
from pathlib import Path

import pandas as pd


def load_incidents(path: str | Path) -> pd.DataFrame:
    """Загружает и подготавливает данные об инцидентах SOC."""

    # 1. Загрузка исходных данных
    df = pd.read_parquet(path)

    # 2. Проверка необходимых столбцов
    required_columns = [
        "EntityId",
        "Создан",
        "Дата закрытия (факт)",
        "Тип инцидента",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Отсутствуют столбцы: {missing_columns}"
        )

    # 3. Проверка идентификаторов
    if df["EntityId"].isna().any():
        raise ValueError("Обнаружены пустые EntityId")

    if df["EntityId"].duplicated().any():
        raise ValueError("Обнаружены дубликаты EntityId")

    # 4. Подготовка дат для расчёта длительности
    created = pd.to_datetime(
        df["Создан"], utc=True, errors="coerce"
    )

    closed = pd.to_datetime(
        df["Дата закрытия (факт)"],
        utc=True,
        errors="coerce",
    )

    duration = (closed - created).dt.total_seconds() / 3600

    # 5. Признак некорректной длительности
    df["invalid_duration"] = duration < 0

    # 6. Неотрицательная длительность в часах
    df["duration_hours"] = duration.where(duration >= 0)

    # 7. Отдельное поле для отображения на графиках
    df["incident_type_display"] = (
        df["Тип инцидента"].fillna("Не указан")
    )

    return df
