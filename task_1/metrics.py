"""Расчёты для четырёх вопросов дашборда. Не зависят от Taipy и Plotly."""

from __future__ import annotations

import numpy as np
import pandas as pd

from loader import DAY_TYPES, FP, NO_VERDICT, OUTCOMES, REAL, SEVERITY_ORDER

TOP_RULES = 10
MIN_SEVERITY_CARDS = 20  # меньше — кривая распределения слишком рваная
MIN_CELL_CARDS = 5  # меньше — доля нарушений в ячейке ничего не значит
HOUR_BLOCK = 3  # ширина окна тепловой карты, часов
INSTANT_SEC = 1.0  # левая граница шкалы времени на графике скорости
QUIET_REAL_SHARE = 0.01  # правило «почти без инцидентов»: реальных меньше 1 % его карточек


# ---------------------------------------------------------------- 1. Поток
def flow_by_time(cards: pd.DataFrame, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, str]:
    """Число карточек по исходам: по дням для коротких окон, по неделям для длинных."""
    if len(dates) == 0:
        return pd.DataFrame(columns=OUTCOMES), "D"
    span = (dates.max() - dates.min()).days + 1
    freq = "D" if span <= 45 else "W"
    if freq == "D":
        bucket = cards["date"]
        index = pd.date_range(dates.min(), dates.max(), freq="D")
    else:
        bucket = cards["date"] - pd.to_timedelta(cards["date"].dt.dayofweek, unit="D")
        first = dates.min() - pd.Timedelta(int(dates.min().dayofweek), unit="D")
        index = pd.date_range(first, dates.max(), freq="7D")
    counts = (
        cards.assign(bucket=bucket)
        .groupby(["bucket", "outcome"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=index, columns=OUTCOMES, fill_value=0)
    )
    return counts, freq


def flow_kpis(cards: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    n = len(cards)
    shares = cards["outcome"].value_counts(normalize=True) if n else pd.Series(dtype=float)
    return {
        "cards": n,
        "days": len(dates),
        "per_day": n / max(len(dates), 1),
        "fp_share": float(shares.get(FP, 0.0)),
        "real": int((cards["outcome"] == REAL).sum()),
        "real_share": float(shares.get(REAL, 0.0)),
        "no_verdict_share": float(shares.get(NO_VERDICT, 0.0)),
    }


# ---------------------------------------------------------------- 2. Правила
def top_rules(cards: pd.DataFrame, n: int = TOP_RULES) -> pd.DataFrame:
    """Самые «громкие» правила: число карточек по исходам, всего и доля потока."""
    if cards.empty:
        return pd.DataFrame(columns=[*OUTCOMES, "total", "share"])
    table = pd.crosstab(cards["rule"], cards["outcome"]).reindex(columns=OUTCOMES, fill_value=0)
    table["total"] = table[OUTCOMES].sum(axis=1)
    table = table.sort_values(["total", REAL], ascending=[False, True]).head(n)
    table["share"] = table["total"] / len(cards)
    return table


def quiet_rules(table: pd.DataFrame) -> pd.DataFrame:
    """Правила топа, у которых реальных инцидентов меньше 1 % карточек — кандидаты на донастройку."""
    if table.empty:
        return table
    return table.loc[table[REAL] < table["total"] * QUIET_REAL_SHARE]


def rules_kpis(cards: pd.DataFrame, n: int = TOP_RULES) -> dict:
    table = top_rules(cards, n)
    quiet = quiet_rules(table)
    return {
        "rules_total": int(cards["rule"].nunique()),
        "top_n": len(table),
        "top_share": float(table["share"].sum()) if len(table) else 0.0,
        "quiet_rules": len(quiet),
        "quiet_share": float(quiet["share"].sum()) if len(quiet) else 0.0,
        "leader": table.index[0] if len(table) else None,
        "leader_share": float(table["share"].iloc[0]) if len(table) else 0.0,
    }


# ---------------------------------------------------------------- 3. Скорость реакции
def reaction_groups(cards: pd.DataFrame) -> dict[str, np.ndarray]:
    """Секунды до взятия в работу по уровням критичности (только уровни с достаточной выборкой)."""
    groups = {}
    for level in SEVERITY_ORDER:
        values = cards.loc[cards["severity"] == level, "react_sec"].dropna().to_numpy()
        if len(values) >= MIN_SEVERITY_CARDS:
            groups[level] = np.sort(values)
    return groups


def ecdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Доля значений, не превышающих каждую точку сетки."""
    return np.searchsorted(values, grid, side="right") / len(values)


def ecdf_grid(groups: dict[str, np.ndarray], points: int = 160) -> np.ndarray:
    top = max(float(v.max()) for v in groups.values())
    return np.geomspace(INSTANT_SEC, max(top, INSTANT_SEC * 10), points)


def reaction_kpis(cards: pd.DataFrame) -> dict:
    react = cards["react_sec"].dropna()
    critical = cards.loc[cards["severity"] == "Критическая", "react_sec"].dropna()
    rest = cards.loc[cards["severity"] != "Критическая", "react_sec"].dropna()
    return {
        "known": len(react),
        # Карточки без даты взятия в работу или с некорректным интервалом.
        "missing": len(cards) - len(react),
        "missing_share": (len(cards) - len(react)) / len(cards) if len(cards) else np.nan,
        "median_all": float(react.median()) if len(react) else np.nan,
        "median_critical": float(critical.median()) if len(critical) else np.nan,
        "median_rest": float(rest.median()) if len(rest) else np.nan,
        "critical_n": len(critical),
        "instant_share": float((react < INSTANT_SEC).mean()) if len(react) else np.nan,
        "medians": {
            level: float(np.median(values)) for level, values in reaction_groups(cards).items()
        },
    }


# ---------------------------------------------------------------- 4. SLA
def hour_blocks() -> list[str]:
    return [f"{h}–{h + HOUR_BLOCK}" for h in range(0, 24, HOUR_BLOCK)]


def headline_min_cards(known: int) -> int:
    """Сколько карточек должно быть в ячейке, чтобы назвать её «худшей» в плитке и отчёте.

    На полном наборе — 100: иначе заголовок дают праздничные ячейки из пары десятков карточек.
    На выборке из 1 000 карточек — 2 % карточек с известным временем, но не меньше 5.
    """
    return max(MIN_CELL_CARDS, min(100, known // 50))


def sla_matrix(cards: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Доля нарушений SLA взятия в работу и число карточек: тип дня × окно суток."""
    # Строки — только типы дней, попавшие в выборку (фильтр «Рабочие дни» убирает выходные).
    present = set(cards["day_type"]) if "day_type" in cards else set(DAY_TYPES)
    rows = [label for code, label in DAY_TYPES.items() if code in present] or list(DAY_TYPES.values())
    cols = hour_blocks()
    known = cards.loc[cards["sla_breach"].notna()]
    if known.empty:
        empty = pd.DataFrame(np.nan, index=rows, columns=cols)
        return empty, empty.fillna(0)
    frame = pd.DataFrame(
        {
            "row": known["day_type"].map(DAY_TYPES),
            "col": known["hour"].floordiv(HOUR_BLOCK).map(dict(enumerate(cols))),
            "breach": known["sla_breach"].astype(float),
        }
    )
    grouped = frame.groupby(["row", "col"])["breach"]
    counts = grouped.size().unstack().reindex(index=rows, columns=cols).fillna(0)
    share = grouped.mean().unstack().reindex(index=rows, columns=cols)
    share = share.where(counts >= MIN_CELL_CARDS)
    return share, counts


def sla_kpis(cards: pd.DataFrame) -> dict:
    known = cards["sla_breach"].dropna()
    share, counts = sla_matrix(cards)
    min_cards = headline_min_cards(len(known))
    worst = share.where(counts >= min_cards).stack().sort_values(ascending=False)
    worst_cell = worst.index[0] if len(worst) else None
    return {
        "known": len(known),
        "breach_share": float(known.astype(float).mean()) if len(known) else np.nan,
        "breaches": int(known.sum()) if len(known) else 0,
        "headline_min": min_cards,
        "worst_cell": worst_cell,
        "worst_share": float(worst.iloc[0]) if len(worst) else np.nan,
        "worst_cards": int(counts.loc[worst_cell]) if worst_cell is not None else 0,
        "best_share": float(worst.iloc[-1]) if len(worst) else np.nan,
    }
