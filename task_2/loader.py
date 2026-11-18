"""Загрузка данных задания 2: лог работы модели и история потока карточек.

Лог модели (MODEL_PREDICTIONS_target.parquet) уже содержит поля карточки, поэтому
отдельный файл карточек не нужен. Основной датасет курса (iavd_ml_dataset.parquet, ≈300 МБ)
нужен только для вопроса 5 «изменился ли поток»: из него читаются 4 столбца.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PREDICTIONS_FILE = DATA_DIR / "MODEL_PREDICTIONS_target.parquet"
DATASET_FILE = DATA_DIR / "iavd_ml_dataset.parquet"

TZ = "Europe/Moscow"
THRESHOLD = 0.18  # решение модели: оценка ≥ 0,18 — «инцидент» (1 − порог 0,82 класса FP)
# Уровни риска модели (шкала Харрингтона из справочника): границы по оценке «инцидент».
RISK_LEVELS = {1: "Низкий", 2: "Средний", 3: "Высокий", 4: "Критический"}
RISK_BOUNDS = (0.37, 0.63, 0.80)
# Окно обучения модели неизвестно. Версия называется wlarf-dec-jun-v3, поэтому считаем
# окном обучения декабрь 2024 — июнь 2025 и везде подписываем это как предположение.
TRAIN_MONTHS = ("2024-12", "2025-06")

SEVERITY_ORDER = ["Критическая", "Высокая", "Средняя", "Низкая", "Отсутствует"]
MONTH_NAMES = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь",
               "октябрь", "ноябрь", "декабрь"]

PREDICTION_FIELDS = {
    "incident_id": "id",
    "timestamp": "assessed",
    "y": "label",
    "y_pred": "predicted",
    "true_incident_proba": "score",
    "risk_level": "risk",
    "Критичность": "severity",
    "Пиктограмма IRP": "category",
    "Вендор из сием": "vendor",
}
DATASET_FIELDS = ["Создан", "Вердикт", "Критичность", "Вендор из сием"]

CATEGORY_FIXES = {"Атака на инфраструтуру": "Атака на инфраструктуру", "Управление УЗ": "Управление учётными записями"}


@dataclass(frozen=True)
class Data:
    predictions: pd.DataFrame  # только карточки с вердиктом «Решён» / «False Positive»
    excluded: int  # оценки без пригодного вердикта (спам, дубликаты, без карточки)
    history: pd.DataFrame | None  # месяц × критичность: карточки, инциденты, FP, вендор
    path: Path


@dataclass(frozen=True)
class Filters:
    month: str = "Весь период"
    severities: tuple[str, ...] = ()


def month_label(month: str) -> str:
    period = pd.Period(month, freq="M")
    return f"{MONTH_NAMES[period.month - 1].capitalize()} {period.year}"


def filled(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.notna() & text.ne("") & text.ne("[]")


def prepare_predictions(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    frame = raw.copy()
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("incident_id должен быть заполнен и уникален.")
    if not frame["score"].between(0, 1).all():
        raise ValueError("Оценка модели должна лежать в [0, 1].")
    if not frame["predicted"].eq(frame["score"].ge(THRESHOLD).astype(int)).all():
        raise ValueError("Решения модели не совпадают с порогом 0,18 из справочника.")
    labelled = frame["label"].isin([0, 1])
    excluded = int((~labelled).sum())
    frame = frame.loc[labelled].copy()
    frame["label"] = frame["label"].astype(int)
    local = pd.to_datetime(frame["assessed"], utc=True).dt.tz_convert(TZ)
    frame["month"] = local.dt.strftime("%Y-%m")
    day = local.dt.tz_localize(None).dt.normalize()
    frame["week"] = day - pd.to_timedelta(day.dt.dayofweek, unit="D")
    frame["severity"] = frame["severity"].fillna("Отсутствует")
    frame["category"] = frame["category"].fillna("Не указано").replace(CATEGORY_FIXES)
    frame["risk_name"] = frame["risk"].map(RISK_LEVELS)
    frame["vendor_filled"] = filled(frame["vendor"])
    frame["outcome"] = np.select(
        [frame["label"].eq(1) & frame["predicted"].eq(1), frame["label"].eq(1) & frame["predicted"].eq(0),
         frame["label"].eq(0) & frame["predicted"].eq(1)],
        ["TP", "FN", "FP"], "TN")
    return frame.reset_index(drop=True), excluded


def prepare_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Сырые 4 столбца основного датасета -> месяц × критичность (поток до и во время работы модели)."""
    created = pd.to_datetime(raw["Создан"], errors="coerce", utc=True, format="mixed")
    frame = pd.DataFrame({
        "month": created.dt.tz_convert(TZ).dt.strftime("%Y-%m"),
        "severity": raw["Критичность"].fillna("Отсутствует"),
        "real": raw["Вердикт"].isin(["Решён", "Решен"]),
        "fp": raw["Вердикт"].eq("False Positive"),
        "vendor_filled": filled(raw["Вендор из сием"]),
    }).loc[created.notna()]
    return (frame.groupby(["month", "severity"])
            .agg(cards=("real", "size"), real=("real", "sum"), fp=("fp", "sum"), vendor=("vendor_filled", "sum"))
            .reset_index())


def load(predictions_path: Path | None = None, dataset_path: Path | None = None) -> Data:
    predictions_path = Path(predictions_path or PREDICTIONS_FILE)
    if not predictions_path.exists():
        raise SystemExit(f"Не найден {predictions_path}. Положите MODEL_PREDICTIONS_target.parquet в папку data/.")
    raw = pd.read_parquet(predictions_path, columns=list(PREDICTION_FIELDS)).rename(columns=PREDICTION_FIELDS)
    predictions, excluded = prepare_predictions(raw)
    dataset_path = Path(dataset_path or DATASET_FILE)
    history = None
    if dataset_path.exists():
        history = prepare_history(pd.read_parquet(dataset_path, columns=DATASET_FIELDS))
    return Data(predictions=predictions, excluded=excluded, history=history, path=predictions_path)


def apply_filters(frame: pd.DataFrame, f: Filters, month_column: str = "month") -> pd.DataFrame:
    selected = frame
    if f.month != "Весь период":
        selected = selected.loc[selected[month_column] == f.month]
    if f.severities:
        selected = selected.loc[selected["severity"].isin(f.severities)]
    return selected


def ru_int(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", " ")


def ru_num(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def ru_pct(share: float, digits: int = 1) -> str:
    if share is None or not np.isfinite(share):
        return "—"
    return f"{ru_num(share * 100, digits)} %"
