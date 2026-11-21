"""Сборка всего, что показывает дашборд задания 3, для одного набора фильтров.

`compute` используют и приложение, и отчёт, поэтому числа на плитках и в REPORT.md совпадают.
"""

from __future__ import annotations

import numpy as np

import charts
import metrics
from loader import (ENRICHMENT, OUTCOME_LABELS, PROJECT_TEAM, THRESHOLD,
                    RECONSTRUCTION_TOLERANCE, Data, Filters, apply_filters, ru_int, ru_pct, ru_proba)

DASH = "—"

TILES = {
    "t_card": "Выбранная карточка",
    "t_check": "Объяснение сходится с оценкой",
    "t_core": "Главная опора группы",
    "t_doubt": "Условное снижение ниже порога",
}


def tile(key: str, value: str, note: str = "", alert: bool = False) -> dict:
    return {f"{key}_v": value, f"{key}_n": note, f"{key}_c": "kpi" + (" kpi-alert" if alert else "")}


def caption(data: Data, f: Filters) -> str:
    shown = len(apply_filters(data.shap, f))
    return (f"Команда SHAP № {PROJECT_TEAM} · модель lgbm_gate1__v001, порог {THRESHOLD} · "
            "SHAP в логарифме шансов объясняет решение модели, а не причину инцидента · "
            f"показано: {ru_int(shown)} из {ru_int(len(data.cards))} карточек группы")


def card_label(row) -> str:
    return f"{row.EntityId} · {row.outcome} · p={row.proba}"


def compute(data: Data, f: Filters, entity_id: int | None = None) -> dict:
    flow = apply_filters(data.shap, f)
    out: dict = {"caption": caption(data, f)}

    # Уровень 1: одна карточка — водопад решения.
    row = metrics.card_row(flow, entity_id) if entity_id else None
    if row is None:
        row = flow.iloc[0] if len(flow) else None
    if row is not None:
        wf = metrics.waterfall_values(row, data.features, data.base)
        out["fig_card"] = charts.waterfall(wf, float(row["proba"]))
        outcome_short = {"TP": "инцидент", "FN": "инцидент", "FP": "не инцидент", "TN": "не инцидент"}[row["outcome"]]
        decision = "модель подняла в приоритет" if row["proba"] >= THRESHOLD else "модель отложила"
        out.update(tile("t_card", ru_proba(float(row["proba"])),
                        f"карточка {row['EntityId']} · истина: {outcome_short} → {decision}",
                        alert=row["outcome"] in ("FN", "FP")))
    else:
        out["fig_card"] = charts.empty()
        out.update(tile("t_card", DASH, "нет карточек"))

    # Уровень 1½: проверяемость — вклады сходятся с оценкой модели.
    worst = max(data.check, 0.0)
    note = (f"Сверено с raw_score (Δ ≤ {data.raw_check:.1e}) и proba"
            if data.raw_check is not None else
            "Сверено с округлённой proba; raw_score отсутствует")
    out.update(tile("t_check", "да" if worst <= RECONSTRUCTION_TOLERANCE else f"±{worst:.1e}",
                    note, alert=worst > RECONSTRUCTION_TOLERANCE))

    # Уровень 2: поток — на чём модель держится.
    top = metrics.mean_abs(flow, data.features)
    out["fig_flow"] = charts.mean_abs(top)
    if len(top):
        head = top.iloc[0]
        enrichment = int(head["feature"] in ENRICHMENT)
        out.update(tile("t_core", str(head["feature"]),
                        f"средний вклад {head['mean_abs']:.2f} лог. шанса — "
                        f"{ru_pct(head['share'], 0)} суммы топ-10",
                        alert=bool(enrichment)))
    else:
        out.update(tile("t_core", DASH, "нет карточек"))

    # Уровень 2: куда толкает признак у разных исходов.
    out["fig_outcomes"] = charts.signed_outcomes(metrics.signed_by_outcome(flow, data.features))

    # Уровень 3: проверяем чувствительность объяснения к выбранным SHAP-вкладам.
    doubt = metrics.enrichment_doubt(flow, data.features)
    doubtful = metrics.doubtful_case(flow, data.features)
    out["doubt_caption"] = (
        f"Пример FP: карточка № {int(doubtful['EntityId'])}. "
        "Выбор карточки для водопада на этот график не влияет."
        if doubtful is not None else "В выбранных исходах нет подходящей FP-карточки."
    )
    out["fig_alert"] = (
        charts.doubt_comparison(
            float(doubtful["proba"]),
            metrics.conditional_without_enrichment(doubtful, data.features),
            int(doubtful["EntityId"]),
        ) if doubtful is not None else charts.empty("Нет подходящих ложных тревог")
    )
    if doubt["raised"]:
        alert = doubt["share"] >= 0.3
        note = (f"у {ru_int(doubt['on_enrichment'])} из {ru_int(doubt['raised'])} поднятых "
                "условная оценка ниже порога; среди FP с оценкой ниже 10% — "
                f"{ru_int(doubt['fp_near'])} из {ru_int(doubt['fp_near_total'])}")
        out.update(tile("t_doubt", ru_pct(doubt["share"], 0) if np.isfinite(doubt["share"]) else DASH,
                        note, alert=alert))
    else:
        out.update(tile("t_doubt", DASH, "модель не подняла ни одной карточки"))
    return out


def card_labels(data: Data, f: Filters) -> list[str]:
    cards = metrics.filtered_cards(data, f)
    return [f"{r.EntityId} · {r.outcome} · p={r.proba}" for r in cards.itertuples()]


def entity_of(label: str) -> int:
    return int(label.split("·")[0].strip())


def summary(data: Data, f: Filters, entity_id: int) -> dict:
    """Сырые числа для отчёта."""
    flow = apply_filters(data.shap, f)
    row = metrics.card_row(data.shap, entity_id)
    return {
        "flow": flow,
        "top": metrics.mean_abs(flow, data.features),
        "signed": metrics.signed_by_outcome(flow, data.features),
        "doubt": metrics.enrichment_doubt(flow, data.features),
        "confirmed": metrics.confirmed_case(flow, data.features),
        "doubtful": metrics.doubtful_case(flow, data.features),
        "card": None if row is None else {
            "contribs": metrics.card_contributions(row, data.features),
            "waterfall": metrics.waterfall_values(row, data.features, data.base),
            "proba": float(row["proba"]),
            "entity": entity_id,
            "outcome": row["outcome"],
        },
    }
