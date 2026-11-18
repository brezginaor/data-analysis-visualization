"""Сборка всего, что показывает дашборд: KPI-плитки и фигуры для одного набора фильтров.

Одна функция `compute` нужна и приложению, и отчёту, поэтому числа на плитках
и в REPORT.md всегда совпадают.
"""

from __future__ import annotations

import numpy as np

import charts
import metrics
from loader import FULL_FILE, Dataset, Filters, apply_filters, ru_duration, ru_int, ru_num, ru_pct, window_dates

DASH = "—"
SHORT_DAY = {"Пн–чт": "пн–чт", "Пятница и предпраздничные": "пятница", "Выходные": "выходные",
             "Праздники": "праздники"}

# Подписи плиток: ключ -> подпись. Порядок — порядок на странице.
TILES = {
    "t_cards": "Карточек",
    "t_fp": "Ложные срабатывания",
    "t_real": "Реальные инциденты",
    "t_rules": "Доля топ-10 правил",
    "t_react": "Взятие критических",
    "t_sla": "Нарушения SLA взятия",
}


def tile(key: str, value: str, note: str = "", alert: bool = False) -> dict:
    return {f"{key}_v": value, f"{key}_n": note, f"{key}_c": "kpi kpi-alert" if alert else "kpi"}


def dataset_label(data: Dataset) -> str:
    kind = "Полный набор" if data.path.name == FULL_FILE.name else f"Выборка {ru_int(len(data.cards))} карточек"
    return f"{kind} · {data.first_date:%d.%m.%Y} — {data.last_date:%d.%m.%Y}"


def compute(data: Dataset, f: Filters) -> dict:
    cards = apply_filters(data, f)
    dates = window_dates(data, f)
    out: dict = {}
    window = f"{dates.min():%d.%m.%Y} — {dates.max():%d.%m.%Y}" if len(dates) else "нет дат"
    out["caption"] = f"{dataset_label(data)} · окно: {window}"

    # 1. Поток
    flow = metrics.flow_kpis(cards, dates)
    counts, freq = metrics.flow_by_time(cards, dates)
    out["fig_flow"] = charts.flow(counts, freq)
    out["flow_note"] = "по дням" if freq == "D" else "по неделям"
    if flow["cards"]:
        out.update(tile("t_cards", ru_int(flow["cards"]),
                        f"≈ {ru_num(flow['per_day'], 0 if flow['per_day'] >= 10 else 1)} в день · {ru_int(flow['days'])} дн."))
        out.update(tile("t_fp", ru_pct(flow["fp_share"]), f"без вердикта ещё {ru_pct(flow['no_verdict_share'])}"))
        ratio = flow["fp_share"] / flow["real_share"] if flow["real_share"] else np.nan
        note = f"{ru_pct(flow['real_share'])} потока" + (
            f" · {ru_num(ratio)} FP на один" if np.isfinite(ratio) else "")
        out.update(tile("t_real", ru_int(flow["real"]), note))
    else:
        for key in ("t_cards", "t_fp", "t_real"):
            out.update(tile(key, DASH, "нет карточек"))

    # 2. Правила
    table = metrics.top_rules(cards)
    rules = metrics.rules_kpis(cards)
    out["fig_rules"] = charts.rules(table)
    if rules["top_n"]:
        note = (f"{rules['quiet_rules']} из {rules['top_n']} почти без инцидентов: "
                f"{ru_pct(rules['quiet_share'])} потока" if rules["quiet_rules"] else
                "у каждого есть реальные инциденты")
        out.update(tile("t_rules", ru_pct(rules["top_share"]), note, alert=rules["quiet_share"] >= 0.05))
    else:
        out.update(tile("t_rules", DASH, "нет карточек"))

    # 3. Скорость реакции
    groups = metrics.reaction_groups(cards)
    grid = metrics.ecdf_grid(groups) if groups else np.array([])
    react = metrics.reaction_kpis(cards)
    out["fig_react"] = (charts.reaction(groups, grid, metrics.ecdf) if groups else charts.empty(
        f"Мало карточек с датой взятия в работу: {ru_int(react['known'])}"))
    if np.isfinite(react["median_critical"]):
        slower = np.isfinite(react["median_rest"]) and react["median_critical"] > react["median_rest"]
        rest = (f"медиана; остальные — {ru_duration(react['median_rest'])}"
                if np.isfinite(react["median_rest"]) else "медиана")
        out.update(tile("t_react", ru_duration(react["median_critical"]), rest, alert=slower))
    else:
        out.update(tile("t_react", DASH, "нет критических карточек"))

    # 4. SLA
    share, cell_counts = metrics.sla_matrix(cards)
    sla = metrics.sla_kpis(cards)
    out["fig_sla"] = (charts.sla(share, cell_counts) if share.notna().any().any() else charts.empty(
        f"Мало карточек с известным временем взятия: {ru_int(sla['known'])}"))
    if sla["known"]:
        note = f"из {ru_int(sla['known'])} с известным временем"
        if sla["worst_cell"] is not None:
            row, col = sla["worst_cell"]
            note = f"хуже всего: {SHORT_DAY.get(row, row)}, {col} ч — {ru_pct(sla['worst_share'], 0)}"
        out.update(tile("t_sla", ru_pct(sla["breach_share"]), note, alert=sla["breach_share"] >= 0.1))
    else:
        out.update(tile("t_sla", DASH, "нет данных о взятии в работу"))
    return out


def summary(data: Dataset, f: Filters) -> dict:
    """Сырые числа для отчёта (без форматирования)."""
    cards = apply_filters(data, f)
    dates = window_dates(data, f)
    return {
        "flow": metrics.flow_kpis(cards, dates),
        "rules": metrics.rules_kpis(cards),
        "top_rules": metrics.top_rules(cards),
        "react": metrics.reaction_kpis(cards),
        "sla": metrics.sla_kpis(cards),
        "sla_matrix": metrics.sla_matrix(cards),
    }
