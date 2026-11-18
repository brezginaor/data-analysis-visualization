"""Сборка всего, что показывает дашборд задания 2, для одного набора фильтров.

`compute` используют и приложение, и отчёт, поэтому числа на плитках и в REPORT.md совпадают.
"""

from __future__ import annotations

import numpy as np

import charts
import metrics
from loader import TRAIN_MONTHS, Data, Filters, apply_filters, month_label, ru_int, ru_pct

DASH = "—"

TILES = {
    "t_miss": "Пропущено инцидентов",
    "t_defer": "Модель отложила",
    "t_prec": "Тревоги модели подтвердились",
    "t_blind": "Слепая зона",
    "t_risk": "Инциденты с риском «низкий»",
    "t_flow": "Доля инцидентов в потоке",
}
TEXT_TILES = {"t_blind"}


def tile(key: str, value: str, note: str = "", alert: bool = False) -> dict:
    kind = " kpi-text" if key in TEXT_TILES else ""
    return {f"{key}_v": value, f"{key}_n": note, f"{key}_c": "kpi" + kind + (" kpi-alert" if alert else "")}


def work_months(frame) -> tuple[str, str] | None:
    if frame.empty:
        return None
    return frame["month"].min(), frame["month"].max()


def caption(data: Data, f: Filters) -> str:
    p = data.predictions
    span = f"{month_label(p['month'].min())} — {month_label(p['month'].max())}".lower()
    window = "весь период" if f.month == "Весь период" else month_label(f.month).lower()
    return (f"Модель wlarf-dec-jun-v3, порог 0,18 · оценки: {span} · с вердиктом: {ru_int(len(p))}, "
            f"без вердикта: {ru_int(data.excluded)} (в метрики не входят) · окно: {window}")


def compute(data: Data, f: Filters) -> dict:
    frame = apply_filters(data.predictions, f)
    out: dict = {"caption": caption(data, f)}
    c = metrics.confusion(frame)

    # 1. Пропуски по неделям
    out["fig_weekly"] = charts.weekly(metrics.weekly_miss(frame), c["miss"])
    if c["incidents"]:
        out.update(tile("t_miss", ru_pct(c["miss"]),
                        f"{ru_int(c['FN'])} из {ru_int(c['incidents'])} · интервал "
                        f"{c['miss_low'] * 100:.0f}–{c['miss_high'] * 100:.0f} %", alert=c["miss"] >= 0.1))
    else:
        out.update(tile("t_miss", DASH, "нет реальных инцидентов"))

    # 2. Кривая поиска
    curve = metrics.gains(frame)
    out["fig_gains"] = charts.gains(curve)
    if c["n"]:
        out.update(tile("t_defer", ru_pct(c["deferred"]),
                        f"очереди: {ru_int(c['FN'] + c['TN'])} из {ru_int(c['n'])} карточек"))
        if c["alerts"]:
            every = int(round(1 / c["precision"])) if c["precision"] > 0 else None
            note = f"{ru_int(c['TP'])} из {ru_int(c['alerts'])}" + (f" — верна каждая {every}-я" if every else "")
            out.update(tile("t_prec", ru_pct(c["precision"]), note, alert=c["precision"] < 0.5))
        else:
            out.update(tile("t_prec", DASH, "модель не подняла ни одной карточки"))
    else:
        out.update(tile("t_defer", DASH, "нет карточек"))
        out.update(tile("t_prec", DASH, "нет карточек"))

    # 3. Слепые зоны
    table = metrics.category_miss(frame)
    out["fig_categories"] = charts.categories(table, c["miss"])
    if len(table):
        worst = table.iloc[0]
        out.update(tile("t_blind", table.index[0],
                        f"пропущено {ru_int(worst['missed'])} из {ru_int(worst['incidents'])} "
                        f"({ru_pct(worst['share'], 0)})", alert=worst["share"] >= 0.5))
    else:
        out.update(tile("t_blind", DASH, "мало инцидентов"))

    # 4. Уровень риска
    rows = metrics.risk_rows(frame)
    out["fig_risk"] = charts.risk(rows)
    low_real = rows.loc[rows["row"].str.startswith("Низкий"), "real"].sum()
    if c["incidents"]:
        out.update(tile("t_risk", ru_pct(low_real / c["incidents"]),
                        f"{ru_int(low_real)} из {ru_int(c['incidents'])}; здесь же все пропуски",
                        alert=low_real / c["incidents"] >= 0.2))
    else:
        out.update(tile("t_risk", DASH, "нет реальных инцидентов"))

    # 5. Изменился ли поток
    monthly = metrics.history_monthly(data.history, f.severities)
    months = work_months(frame) or work_months(data.predictions)
    out["fig_history"] = charts.history(monthly, TRAIN_MONTHS, months)
    if not monthly.empty and months:
        cmp = metrics.train_vs_work(monthly, months)
        out.update(tile("t_flow", ru_pct(cmp["work"]["real_share"]),
                        f"при обучении — {ru_pct(cmp['train']['real_share'])}; «Вендор»: "
                        f"{ru_pct(cmp['train']['vendor_share'], 0)} → {ru_pct(cmp['work']['vendor_share'], 0)}",
                        alert=abs(cmp["work"]["real_share"] - cmp["train"]["real_share"]) >= 0.03))
    else:
        share = frame["label"].mean() if len(frame) else np.nan
        out.update(tile("t_flow", ru_pct(share), "нет iavd_ml_dataset.parquet для сравнения"))
    return out


def summary(data: Data, f: Filters) -> dict:
    """Сырые числа для отчёта."""
    frame = apply_filters(data.predictions, f)
    monthly = metrics.history_monthly(data.history, f.severities)
    months = work_months(frame)
    return {
        "confusion": metrics.confusion(frame),
        "monthly": metrics.monthly_miss(frame),
        "weekly": metrics.weekly_miss(frame),
        "gains": metrics.gains(frame),
        "queue_for_90": metrics.queue_share_for(frame, 0.9),
        "categories": metrics.category_miss(frame),
        "risk": metrics.risk_rows(frame),
        "history": monthly,
        "compare": metrics.train_vs_work(monthly, months) if (not monthly.empty and months) else None,
        "work_months": months,
    }
