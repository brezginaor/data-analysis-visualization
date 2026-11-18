"""Расчёты для пяти вопросов о модели. Не зависят от Taipy и Plotly.

Все метрики качества считаются только по карточкам с вердиктом «Решён» (инцидент)
или «False Positive»: остальные сверить с моделью не с чем.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from loader import RISK_BOUNDS, THRESHOLD, TRAIN_MONTHS

MIN_WEEK_INCIDENTS = 10  # меньше — доля пропусков за неделю почти случайна
MIN_CATEGORY_INCIDENTS = 10


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """95-процентный интервал Уилсона для доли k из n."""
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def confusion(frame: pd.DataFrame) -> dict:
    counts = frame["outcome"].value_counts()
    tp, fn, fp, tn = (int(counts.get(k, 0)) for k in ("TP", "FN", "FP", "TN"))
    incidents, alerts, n = tp + fn, tp + fp, tp + fn + fp + tn
    low, high = wilson(fn, incidents)
    return {
        "n": n, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "incidents": incidents, "alerts": alerts,
        "miss": fn / incidents if incidents else np.nan, "miss_low": low, "miss_high": high,
        "precision": tp / alerts if alerts else np.nan,
        "deferred": (fn + tn) / n if n else np.nan,  # доля очереди, которую модель отложила
    }


# ---------------------------------------------------------------- 1. Пропуски во времени
def weekly_miss(frame: pd.DataFrame) -> pd.DataFrame:
    incidents = frame.loc[frame["label"] == 1]
    if incidents.empty:
        return pd.DataFrame(columns=["week", "incidents", "missed", "share", "low", "high"])
    table = incidents.groupby("week").agg(incidents=("label", "size"),
                                          missed=("outcome", lambda s: int((s == "FN").sum()))).reset_index()
    table["share"] = table["missed"] / table["incidents"]
    bounds = [wilson(k, n) for k, n in zip(table["missed"], table["incidents"])]
    table["low"], table["high"] = zip(*bounds)
    table["reliable"] = table["incidents"] >= MIN_WEEK_INCIDENTS
    return table


def monthly_miss(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, part in frame.groupby("month"):
        c = confusion(part)
        rows.append({"month": month, **c})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. Кривая поиска
def gains(frame: pd.DataFrame) -> dict:
    """Разбираем очередь сверху вниз по оценке модели: какая доля инцидентов найдена."""
    if frame.empty or frame["label"].sum() == 0:
        return {}
    ordered = frame.sort_values("score", ascending=False)
    found = ordered["label"].cumsum().to_numpy() / ordered["label"].sum()
    queue = np.arange(1, len(ordered) + 1) / len(ordered)
    step = max(1, len(ordered) // 400)  # 400 точек достаточно для гладкой кривой
    index = np.r_[np.arange(0, len(ordered), step), len(ordered) - 1]
    at_threshold = int((ordered["score"] >= THRESHOLD).sum())
    point = (at_threshold / len(ordered), found[at_threshold - 1] if at_threshold else 0.0)
    return {"queue": np.r_[0, queue[index]], "found": np.r_[0, found[index]], "threshold": point,
            "base_rate": float(frame["label"].mean())}


def queue_share_for(frame: pd.DataFrame, target: float) -> float:
    """Какую долю очереди нужно разобрать, чтобы найти target инцидентов."""
    curve = gains(frame)
    if not curve:
        return np.nan
    position = np.searchsorted(curve["found"], target)
    return float(curve["queue"][min(position, len(curve["queue"]) - 1)])


# ---------------------------------------------------------------- 3. Слепые зоны
def category_miss(frame: pd.DataFrame, min_incidents: int = MIN_CATEGORY_INCIDENTS) -> pd.DataFrame:
    incidents = frame.loc[frame["label"] == 1]
    table = incidents.groupby("category").agg(incidents=("label", "size"),
                                              missed=("outcome", lambda s: int((s == "FN").sum())))
    table = table.loc[table["incidents"] >= min_incidents].copy()
    if table.empty:
        return table.assign(share=[], low=[], high=[])
    table["share"] = table["missed"] / table["incidents"]
    bounds = [wilson(k, n) for k, n in zip(table["missed"], table["incidents"])]
    table["low"], table["high"] = zip(*bounds)
    return table.sort_values(["share", "incidents"], ascending=[False, False])


# ---------------------------------------------------------------- 4. Уровень риска
def risk_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Доля реальных инцидентов по уровню риска. «Низкий» делится порогом 0,18 на две строки."""
    low, mid, high = RISK_BOUNDS
    rows = [
        ("Низкий, оценка < 0,18 — модель отложила", frame["score"] < THRESHOLD),
        ("Низкий, оценка 0,18–0,37 — модель подняла", frame["score"].between(THRESHOLD, low, inclusive="left")),
        ("Средний (0,37–0,63)", frame["score"].between(low, mid, inclusive="left")),
        ("Высокий (0,63–0,80)", frame["score"].between(mid, high, inclusive="left")),
        ("Критический (от 0,80)", frame["score"] >= high),
    ]
    total_real = int(frame["label"].sum())
    out = []
    for name, mask in rows:
        part = frame.loc[mask]
        real = int(part["label"].sum())
        out.append({"row": name, "cards": len(part), "real": real,
                    "share": real / len(part) if len(part) else np.nan,
                    "of_all_real": real / total_real if total_real else np.nan,
                    "deferred": name.startswith("Низкий, оценка <")})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 5. Изменился ли поток
def history_monthly(history: pd.DataFrame | None, severities: tuple[str, ...] = ()) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame(columns=["month", "cards", "real_share", "vendor_share"])
    part = history if not severities else history.loc[history["severity"].isin(severities)]
    monthly = part.groupby("month")[["cards", "real", "fp", "vendor"]].sum().reset_index()
    labelled = monthly["real"] + monthly["fp"]
    monthly["real_share"] = np.where(labelled > 0, monthly["real"] / labelled.where(labelled > 0, 1), np.nan)
    monthly["vendor_share"] = monthly["vendor"] / monthly["cards"]
    return monthly


def period_summary(monthly: pd.DataFrame, months: tuple[str, str]) -> dict:
    part = monthly.loc[monthly["month"].between(*months)]
    labelled = part["real"].sum() + part["fp"].sum()
    return {
        "real_share": part["real"].sum() / labelled if labelled else np.nan,
        "vendor_share": part["vendor"].sum() / part["cards"].sum() if part["cards"].sum() else np.nan,
        "cards": int(part["cards"].sum()),
    }


def train_vs_work(monthly: pd.DataFrame, work_months: tuple[str, str]) -> dict:
    return {"train": period_summary(monthly, TRAIN_MONTHS), "work": period_summary(monthly, work_months)}
