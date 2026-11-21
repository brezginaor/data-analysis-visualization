# Анализ данных и визуализация

В репозитории находятся три задания курса: анализ карточек SOC, исследование решений модели и SHAP-объяснения.

## Подготовка проекта к работе

Для работы необходимы Python 3.12, Poetry и Git.

**0. Если Python 3.12 или Poetry ещё не установлены**

`poetry env use 3.12` предполагает, что интерпретатор `python3.12` уже доступен в PATH. Варианты установки:

- Python 3.12 — установщик с [python.org/downloads](https://www.python.org/downloads/) или `uv python install 3.12` ([astral.sh/uv](https://docs.astral.sh/uv/getting-started/install/));
- Poetry — [install.pythonpoetry.org](https://python-poetry.org/docs/#installation), например `curl -sSL https://install.pythonpoetry.org | python3 -`.

**1. Клонирование репозитория**

```powershell
git clone https://github.com/brezginaor/data-analysis-visualization.git
cd data-analysis-visualization
```

**2. Установка зависимостей**

```powershell
poetry env use 3.12
poetry install
```

**3. Загрузка данных**

Для заданий 1 и 2 скачайте файлы из [общей папки на Яндекс Диске](https://disk.yandex.ru/d/waiZmiOzun8EZA) и поместите в `data/`. Для задания 3 данные уже включены в репозиторий.

Исходные Parquet-файлы для заданий 1 и 2 хранятся локально. Два CSV для задания 3 уже включены в репозиторий в папке `data/task3/`.

| Файл | Папка на Диске | Нужен для |
|---|---|---|
| `iavd_incidents_raw_target.parquet` | `1.5 Пр Декомпозиция/Dataset/` или `Datasets/Задание 1: Датасет карточек сок + календарь` | основной набор, 240 295 карточек: загрузчик, ноутбук, дашборд задания 1 |
| `iavd_1000_incidents_raw_target.parquet` | там же | учебная выборка: дашборд задания 1 работает и на ней |
| `calendar.parquet` | там же | тип дня в задании 1; без файла тип дня определяется по дню недели |
| `MODEL_PREDICTIONS_target.parquet` | `Datasets/Задание 2: Инференс модели/Готовые данные` | дашборд задания 2 (обязателен) |
| `iavd_ml_dataset.parquet` (≈365 МБ) | `Datasets/Задание 2: Инференс модели/Под обучение` | вопрос 5 задания 2; без файла на месте графика будет подсказка |

**4. Запуск**

Команды выполняются из корня репозитория.

| Команда | Что делает |
|---|---|
| `poetry run python task_1/app.py` | дашборд задания 1 → http://127.0.0.1:5011 |
| `poetry run python task_2/app.py` | дашборд задания 2 → http://127.0.0.1:5013 |
| `poetry run python task_3/app.py` | дашборд задания 3 → http://127.0.0.1:5014 |
| `poetry run python -m unittest discover -s task_1` | проверки расчётов задания 1 (данные не нужны) |
| `poetry run python -m unittest discover -s task_2` | проверки расчётов задания 2 (данные не нужны) |
| `poetry run python -m unittest discover -s task_3` | проверки расчётов задания 3 |
| `poetry run python task_1/report.py` | пересобрать `task_1/REPORT.md` по текущим данным |
| `poetry run python task_2/report.py` | пересобрать `task_2/REPORT.md` по текущим данным |
| `poetry run python task_3/report.py` | сформировать `task_3/REPORT.md` по восьми карточкам команды № 4 |

Ноутбук `notebooks/01_eda.ipynb` запускается в Jupyter или VS Code с ядром окружения Poetry (`ipykernel` входит в dev-зависимости).

**5. Загрузка данных в Python**

```python
from src.data_loader import load_incidents

df = load_incidents("data/iavd_incidents_raw_target.parquet")
```

Функция возвращает DataFrame с 240 295 строками и 100 столбцами (97 исходных и 3 дополнительных).

## Структура репозитория

| Папка | Что внутри |
|---|---|
| `docs/` | ТЗ проекта (`plan.md`) и отчёт по данным (`data_report.md`) |
| `notebooks/` | исследование данных (EDA) |
| `src/` | общий загрузчик датасета (`data_loader.py`) |
| `task_1/` | задание 1 «Карточки SOC»: дашборд Taipy, тесты, `README.md` и `REPORT.md` |
| `task_2/` | задание 2 «Инференс модели»: дашборд Taipy, тесты, `README.md` и `REPORT.md` |
| `task_3/` | задание 3 «SHAP-объяснение»: дашборд Taipy, тесты, `README.md` и `REPORT.md` |
| `data/task3/` | CSV с учебными карточками и SHAP-вкладами для задания 3 |

Результаты исследования данных и описание ограничений приведены в [отчёте](docs/data_report.md). Выводы дашбордов — в `task_1/REPORT.md`, `task_2/REPORT.md` и `task_3/REPORT.md`: отчёты собираются `report.py` из тех же функций, что и плитки дашбордов, поэтому числа совпадают.
