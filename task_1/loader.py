"""Загрузка карточек SOC для дашборда задания 1.

Читаются только столбцы, которые нужны дашборду, поэтому полный файл
(240 тыс. карточек, 97 столбцов) загружается быстро и занимает мало памяти.
Исходные значения не исправляются: сомнительные интервалы помечаются пропуском.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FULL_FILE = DATA_DIR / "iavd_incidents_raw_target.parquet"
SAMPLE_FILE = DATA_DIR / "iavd_1000_incidents_raw_target.parquet"
CALENDAR_FILE = DATA_DIR / "calendar.parquet"

TZ = "Europe/Moscow"
SEP = "~~||~~"

FIELDS = {
    "EntityId": "id",
    "Создан": "created",
    "Вердикт": "verdict",
    "Критичность": "severity",
    "Наименование": "rule",
    "Источник инцидента": "source",
    "Дата взятия в работу": "assigned",
    "SLA по взятию в работу Атаки": "sla_ms",
}

SEVERITY_ORDER = ["Критическая", "Высокая", "Средняя", "Низкая", "Отсутствует"]

REAL, FP, OTHER, NO_VERDICT = (
    "Реальный инцидент",
    "False Positive",
    "Дубликат и прочие",
    "Без вердикта",
)
OUTCOMES = [REAL, FP, OTHER, NO_VERDICT]
OUTCOME_BY_VERDICT = {
    "Решён": REAL,
    "Решен": REAL,
    "False Positive": FP,
    "Не указан": NO_VERDICT,
}

# Тип дня из calendar.parquet (проверено по дням недели):
# 1 — пн–чт, 2 — пятница и предпраздничный день, 3 — праздник, 4 — выходной.
DAY_TYPES = {1: "Пн–чт", 2: "Пятница и предпраздничные", 4: "Выходные", 3: "Праздники"}
WORKDAY_TYPES = {1, 2}

PERIODS = {"Весь период": None, "180 дней": 180, "90 дней": 90, "30 дней": 30, "7 дней": 7}
DAY_FILTERS = {
    "Все дни": None,
    "Рабочие дни": WORKDAY_TYPES,
    "Выходные и праздники": {3, 4},
}


@dataclass(frozen=True)
class Dataset:
    cards: pd.DataFrame
    day_type: pd.Series  # дата -> тип дня, для всех дат периода данных
    path: Path

    @property
    def first_date(self) -> pd.Timestamp:
        return self.cards["date"].min()

    @property
    def last_date(self) -> pd.Timestamp:
        return self.cards["date"].max()


@dataclass(frozen=True)
class Filters:
    period: str = "Весь период"
    days: str = "Все дни"
    severities: tuple[str, ...] = ()


def default_path() -> Path:
    """Полный файл, если он лежит в data/, иначе учебная выборка из 1 000 карточек."""
    return FULL_FILE if FULL_FILE.exists() else SAMPLE_FILE


def weekday_day_type(dates: pd.DatetimeIndex) -> pd.Series:
    """Тип дня по дню недели — запасной вариант для дат, которых нет в календаре."""
    weekday = pd.Series(dates.dayofweek, index=dates)
    return weekday.map(lambda d: 1 if d < 4 else 2 if d == 4 else 4).astype(int)


def load_day_types(dates: pd.DatetimeIndex, path: Path = CALENDAR_FILE) -> pd.Series:
    fallback = weekday_day_type(dates)
    if not Path(path).exists():
        return fallback
    calendar = pd.read_parquet(path, columns=["Дата", "Тип дня"])
    calendar.index = pd.to_datetime(calendar["Дата"]).dt.normalize()
    known = calendar["Тип дня"].astype(int).reindex(dates)
    return known.fillna(fallback).astype(int)


def outcome_of(verdict: pd.Series) -> pd.Series:
    outcome = verdict.map(OUTCOME_BY_VERDICT).fillna(OTHER)
    outcome = outcome.where(verdict.notna(), NO_VERDICT)
    return pd.Series(pd.Categorical(outcome, categories=OUTCOMES), index=verdict.index)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Переименованные сырые столбцы -> карточки с расчётными полями."""
    cards = raw.copy()
    if cards["id"].isna().any() or cards["id"].duplicated().any():
        raise ValueError("EntityId должен быть заполнен и уникален.")
    if cards["created"].isna().any():
        raise ValueError("У каждой карточки должна быть дата создания.")

    cards["severity"] = cards["severity"].fillna("Отсутствует")
    unknown = set(cards["severity"]) - set(SEVERITY_ORDER)
    if unknown:
        raise ValueError(f"Неизвестные значения критичности: {sorted(unknown)}")
    cards["severity"] = pd.Categorical(cards["severity"], categories=SEVERITY_ORDER, ordered=True)

    cards["rule"] = cards["rule"].fillna("Не указано").str.replace(SEP, " / ", regex=False)
    cards["source"] = cards["source"].fillna("Не указан")
    cards["outcome"] = outcome_of(cards["verdict"])

    created = pd.to_datetime(cards["created"], utc=True)
    assigned = pd.to_datetime(cards["assigned"], utc=True)
    local = created.dt.tz_convert(TZ)
    cards["date"] = local.dt.tz_localize(None).dt.normalize()
    cards["hour"] = local.dt.hour.astype(int)

    # Время до взятия в работу. Отрицательный интервал — ошибка данных, а не «мгновенно».
    seconds = (assigned - created).dt.total_seconds()
    cards["react_sec"] = seconds.where(seconds >= 0)

    # Нарушение SLA взятия в работу: интервал длиннее норматива (норматив хранится в мс).
    sla_ms = pd.to_numeric(cards["sla_ms"], errors="coerce")
    known = cards["react_sec"].notna() & sla_ms.gt(0)
    breach = (cards["react_sec"] * 1000).gt(sla_ms)
    cards["sla_breach"] = breach.where(known).astype("boolean")
    return cards.sort_values("created", kind="stable").reset_index(drop=True)


def load(path: str | Path | None = None, calendar_path: Path = CALENDAR_FILE) -> Dataset:
    path = Path(path) if path else default_path()
    if not path.exists():
        raise SystemExit(
            f"Не найден файл {path}.\n"
            "Положите iavd_incidents_raw_target.parquet или "
            "iavd_1000_incidents_raw_target.parquet в папку data/."
        )
    raw = pd.read_parquet(path, columns=list(FIELDS)).rename(columns=FIELDS)
    cards = prepare(raw)
    dates = pd.date_range(cards["date"].min(), cards["date"].max(), freq="D")
    day_type = load_day_types(dates, calendar_path)
    cards["day_type"] = cards["date"].map(day_type).astype(int)
    return Dataset(cards=cards, day_type=day_type, path=path)


def window_dates(data: Dataset, f: Filters) -> pd.DatetimeIndex:
    """Календарные даты выбранного окна с учётом типа дня (для среднего «в день»)."""
    days = PERIODS[f.period]
    end = data.last_date
    start = data.first_date if days is None else max(data.first_date, end - pd.Timedelta(int(days) - 1, unit="D"))
    dates = pd.date_range(start, end, freq="D")
    allowed = DAY_FILTERS[f.days]
    if allowed is not None:
        dates = dates[data.day_type.reindex(dates).isin(allowed).to_numpy()]
    return dates


def apply_filters(data: Dataset, f: Filters) -> pd.DataFrame:
    cards = data.cards
    dates = window_dates(data, f)
    if len(dates) == 0:
        return cards.iloc[0:0]
    selected = cards.loc[cards["date"].between(dates.min(), dates.max())]
    allowed = DAY_FILTERS[f.days]
    if allowed is not None:
        selected = selected.loc[selected["day_type"].isin(allowed)]
    if f.severities:
        selected = selected.loc[selected["severity"].isin(f.severities)]
    return selected


def ru_int(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", " ")


def ru_num(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def ru_pct(share: float, digits: int = 1) -> str:
    return f"{ru_num(share * 100, digits)} %"


def ru_duration(seconds: float) -> str:
    """Секунды -> «< 1 с», «40 с», «5,2 мин», «3,1 ч», «2,4 сут»."""
    if seconds is None or not np.isfinite(seconds):
        return "—"
    if seconds < 1:
        return "< 1 с"
    if seconds < 60:
        return f"{seconds:.0f} с"
    minutes = seconds / 60
    if minutes < 60:
        return f"{ru_num(minutes)} мин"
    if minutes < 1440:
        return f"{ru_num(minutes / 60)} ч"
    return f"{ru_num(minutes / 1440)} сут"
