"""Расчёты для вопросов задания 3. Не зависят от Taipy и Plotly.

Все вклады — TreeSHAP модели lgbm_gate1__v001 в логарифме шансов: положительный вклад
толкает решение «к инциденту», отрицательный — «к легитимному». Объяснение описывает
модель, а не причину инцидента.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loader import ENRICHMENT, OUTCOMES, Filters, THRESHOLD, apply_filters

TOP_FEATURES = 10       # сколько признаков показываем на уровне потока
TOP_CONTRIBS = 8        # сколько вкладов в водопаде карточки
NEAR_THRESHOLD = 0.1    # «у порога»: оценки до 0,1, где решение модели ещё можно переспорить


def card_row(shap: pd.DataFrame, entity_id: int) -> pd.Series | None:
    part = shap.loc[shap["EntityId"] == entity_id]
    return None if part.empty else part.iloc[0]


def card_contributions(row: pd.Series, features: tuple[str, ...]) -> pd.DataFrame:
    """Вклады одной карточки, отсортированные по модулю."""
    table = pd.DataFrame({"feature": list(features), "shap": [float(row[f]) for f in features]})
    return table.reindex(table["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def waterfall_values(row: pd.Series, features: tuple[str, ...], base: float,
                     k: int = TOP_CONTRIBS) -> pd.DataFrame:
    """Базовая линия + k крупнейших вкладов + «остальные признаки» + итог.

    Итог равен логарифму шансов оценки модели: proba = sigmoid(итог).
    """
    contribs = card_contributions(row, features)
    head = contribs.head(k)
    rest = float(contribs["shap"].iloc[k:].sum())
    total = base + float(contribs["shap"].sum())
    rows = [{"feature": "Базовая линия", "value": base, "measure": "absolute"}]
    for r in head.itertuples():
        rows.append({"feature": r.feature, "value": r.shap, "measure": "relative"})
    if rest:
        rows.append({"feature": f"остальные {len(contribs) - k} признаков", "value": rest, "measure": "relative"})
    rows.append({"feature": "Оценка модели (лог. шансов)", "value": total, "measure": "total"})
    return pd.DataFrame(rows)


def mean_abs(flow: pd.DataFrame, features: tuple[str, ...], k: int = TOP_FEATURES) -> pd.DataFrame:
    """Топ признаков потока по среднему |SHAP|: на чём модель «держится»."""
    if flow.empty:
        return pd.DataFrame(columns=["feature", "mean_abs", "share"])
    table = pd.DataFrame({"feature": list(features),
                          "mean_abs": [flow[list(features)].abs().mean().to_dict()[f] for f in features]})
    table = table.sort_values("mean_abs", ascending=False).head(k).reset_index(drop=True)
    table["share"] = table["mean_abs"] / table["mean_abs"].sum()
    return table


def signed_by_outcome(flow: pd.DataFrame, features: tuple[str, ...], k: int = 6) -> pd.DataFrame:
    """Средний подписанный SHAP топ-признаков по исходам: куда толкает признак у TP/FP/FN/TN."""
    top = mean_abs(flow, features, k)
    if top.empty:
        return pd.DataFrame(columns=["feature", "outcome", "mean_shap"])
    rows = []
    for feature in top["feature"]:
        for outcome, part in flow.groupby("outcome"):
            rows.append({"feature": feature, "outcome": outcome,
                         "mean_shap": float(part[feature].mean())})
    return pd.DataFrame(rows)


def enrichment_doubt(flow: pd.DataFrame, features: tuple[str, ...]) -> dict:
    """Сомнительные объяснения: решение «к инциденту» держится на деградировавшем обогащении.

    Контрфактическая проверка: убираем вклады признаков обогащения и смотрим, упала бы
    карточка ниже порога. Если да — без «сломанного» поля модель карточку бы не подняла.
    """
    empty = {"raised": 0, "on_enrichment": 0, "share": np.nan, "fp_near": 0, "fp_near_total": 0}
    present = [f for f in ENRICHMENT if f in features]
    if flow.empty or not present or "base" not in flow.columns:
        return empty
    raised = flow.loc[flow["proba"] >= THRESHOLD]
    if raised.empty:
        return empty
    total = raised["base"] + raised[list(features)].sum(axis=1)
    without = total - raised[present].sum(axis=1)
    kept_below = 1 / (1 + np.exp(-without.clip(-700, 700))) < THRESHOLD
    on_enrichment = int(kept_below.sum())
    near = kept_below & raised["outcome"].eq("FP") & raised["proba"].lt(NEAR_THRESHOLD)
    return {
        "raised": len(raised),
        "on_enrichment": on_enrichment,
        "share": on_enrichment / len(raised),
        "fp_near": int(near.sum()),
        "fp_near_total": int(raised["outcome"].eq("FP").sum()),
    }


def confirmed_case(flow: pd.DataFrame, features: tuple[str, ...]) -> pd.Series | None:
    """Кандидат в «подтверждённые»: TP, который модель подняла бы и без обогащения."""
    present = [f for f in ENRICHMENT if f in features]
    tps = flow.loc[flow["outcome"] == "TP"]
    if tps.empty or "base" not in tps.columns:
        return None
    total = tps["base"] + tps[list(features)].sum(axis=1)
    without = total - tps[present].sum(axis=1) if present else total
    robust = tps.loc[1 / (1 + np.exp(-without.clip(-700, 700))) >= THRESHOLD]
    if robust.empty:
        return None
    push = robust[list(features)].max(axis=1)
    return robust.loc[push.idxmax()]


def doubtful_case(flow: pd.DataFrame, features: tuple[str, ...]) -> pd.Series | None:
    """Кандидат в «сомнительные»: FP, который падает ниже порога без вклада обогащения."""
    present = [f for f in ENRICHMENT if f in features]
    if not present or flow.empty or "base" not in flow.columns:
        return None
    near = flow.loc[flow["outcome"].eq("FP") & flow["proba"].between(THRESHOLD, NEAR_THRESHOLD, inclusive="left")]
    if near.empty:
        return None
    total = near["base"] + near[list(features)].sum(axis=1)
    without = total - near[present].sum(axis=1)
    falls = 1 / (1 + np.exp(-without.clip(-700, 700))) < THRESHOLD
    candidates = near.loc[falls]
    return None if candidates.empty else candidates.sort_values("proba").iloc[-1]


def filtered_cards(data, f: Filters) -> pd.DataFrame:
    """Карточки выбранной команды и исходов для селектора."""
    cards = data.cards
    if f.team is not None:
        cards = cards.loc[cards["team"] == f.team]
    if f.outcomes and set(f.outcomes) != set(OUTCOMES):
        cards = cards.loc[cards["outcome"].isin(f.outcomes)]
    return cards
