"""Загрузка данных задания 3: карточки команд и предрассчитанные SHAP-вклады модели курса.

Файлы лежат в data/task3/ (в Git не попадают), источник — архив IAVD_3.7_ML_team:
    cards_practice.csv — 320 карточек (40 команд × 8: по 2 исхода TP/FP/FN/TN);
    shap_cards.csv     — вклады 73 признаков (логарифм шансов) + base + proba.

Модель: lgbm_gate1__v001, порог 0,0193. SHAP объясняет решение модели, а не причину инцидента.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "task3"
CARDS_FILE = DATA_DIR / "cards_practice.csv"
SHAP_FILE = DATA_DIR / "shap_cards.csv"

THRESHOLD = 0.0193  # рабочая порог модели, часть модели — не менять (см. model_card)
SERVICE = ("EntityId", "y_true", "proba", "outcome", "base")
OUTCOMES = ("TP", "FP", "FN", "TN")

# Признаки обогащения карточки: с сентября 2025 оно деградирует (model_card),
# поэтому вклады этих полей — главный кандидат на «сомнительное» объяснение.
ENRICHMENT = ("Алерт", "Вендор из сием", "len_Значение Инцидента (группировка)")

OUTCOME_LABELS = {
    "TP": "инцидент — модель подняла",
    "FP": "не инцидент — модель подняла",
    "FN": "инцидент — модель отложила",
    "TN": "не инцидент — модель отложила",
}

# proba в shap_cards.csv округлена до 6 знаков, поэтому reconstruction проверяем
# с допуском в полшага последнего разряда.
RECONSTRUCTION_TOLERANCE = 6e-7


@dataclass(frozen=True)
class Data:
    cards: pd.DataFrame  # EntityId, team, created, y_true, proba, outcome, flag
    shap: pd.DataFrame   # EntityId, team, outcome, proba + вклад 73 признаков + base
    features: tuple[str, ...]
    base: float
    check: float  # максимальное расхождение sigmoid(base + ΣSHAP) с proba из файла
    path: Path


@dataclass(frozen=True)
class Filters:
    team: int | None = None  # None — все команды
    outcomes: tuple[str, ...] = OUTCOMES


def ru_int(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", " ")


def ru_num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def ru_pct(share: float, digits: int = 1) -> str:
    if share is None or not np.isfinite(share):
        return "—"
    return f"{ru_num(share * 100, digits)} %"


SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def ru_proba(value: float) -> str:
    """Оценка модели для человека: 0,0209 или 2·10⁻⁶ для совсем маленьких."""
    if value == 0:
        return "0"
    if value >= 0.001:
        return ru_num(value, 4).rstrip("0").rstrip(",").replace(".", ",")
    exp = int(np.floor(np.log10(value)))
    mantissa = ru_num(value / 10**exp, 1)
    return f"{mantissa}·10{str(exp).translate(SUPER)}"


def _read_cards(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    if frame["EntityId"].duplicated().any():
        raise ValueError("EntityId карточек должен быть уникален.")
    if not frame["outcome"].isin(OUTCOMES).all():
        raise ValueError("Исход карточки должен быть TP, FP, FN или TN.")
    if not (frame["outcome"].isin(["TP", "FN"]) == frame["y_true"].eq(1)).all():
        raise ValueError("y_true не согласуется с исходом: инциденты — это TP и FN.")
    if not (frame["proba"].ge(THRESHOLD) == frame["flag"].eq("инцидент")).all():
        raise ValueError(f"Флаг решения не совпадает с порогом {THRESHOLD}.")
    frame["team"] = frame["team"].astype(int)
    frame["EntityId"] = frame["EntityId"].astype(int)
    return frame


def _read_shap(path: Path, features: tuple[str, ...]) -> tuple[pd.DataFrame, float]:
    frame = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    if frame["EntityId"].duplicated().any():
        raise ValueError("EntityId в shap_cards должен быть уникален.")
    missing = [f for f in ("EntityId", "y_true", "proba", "outcome", *features) if f not in frame.columns]
    if missing:
        raise ValueError(f"В shap_cards нет столбцов: {missing}.")
    if not frame["outcome"].isin(OUTCOMES).all():
        raise ValueError("Исход в shap_cards должен быть TP, FP, FN или TN.")
    base = frame["base"]
    if base.nunique() != 1:
        raise ValueError("Базовая линия SHAP должна быть одной для всех карточек.")
    return frame, float(base.iloc[0])


def reconstruction_check(shap: pd.DataFrame, features: tuple[str, ...], base: float) -> float:
    """Максимальное расхождение sigmoid(base + ΣSHAP) с proba из файла."""
    total = (base + shap[list(features)].sum(axis=1)).clip(-700, 700)
    return float((1 / (1 + np.exp(-total)) - shap["proba"]).abs().max())


def load(cards_path: Path | None = None, shap_path: Path | None = None) -> Data:
    cards_path = Path(cards_path or CARDS_FILE)
    shap_path = Path(shap_path or SHAP_FILE)
    for path in (cards_path, shap_path):
        if not path.exists():
            raise SystemExit(f"Не найден {path}. Распакуйте course_pack из IAVD_3.7_ML_team в data/task3/.")
    cards = _read_cards(cards_path)
    shap_raw, base = _read_shap(shap_path, tuple())  # список признаков определим по файлу
    features = tuple(c for c in shap_raw.columns if c not in SERVICE)
    shap, base = _read_shap(shap_path, features)
    check = reconstruction_check(shap, features, base)
    if check > RECONSTRUCTION_TOLERANCE:
        raise ValueError(f"Вклады не сходятся с оценкой модели: расхождение {check:.2e} > {RECONSTRUCTION_TOLERANCE:.0e}.")
    merged = shap.merge(cards[["EntityId", "team", "created", "flag"]], on="EntityId", how="inner", validate="one_to_one")
    if len(merged) != len(cards):
        raise ValueError("Наборы карточек в cards_practice и shap_cards не совпадают.")
    return Data(cards=cards, shap=merged, features=features, base=base, check=check, path=shap_path)


def apply_filters(shap: pd.DataFrame, f: Filters) -> pd.DataFrame:
    selected = shap
    if f.team is not None:
        selected = selected.loc[selected["team"] == f.team]
    if f.outcomes and set(f.outcomes) != set(OUTCOMES):
        selected = selected.loc[selected["outcome"].isin(f.outcomes)]
    return selected
