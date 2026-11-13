## Подготовка проекта к работе

Для работы необходимы Python 3.12, Poetry и Git.

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

**3. Загрузка датасета**

Скачать файл `iavd_incidents_raw_target.parquet` из [общей папки на Яндекс Диске](https://disk.yandex.ru/d/waiZmiOzun8EZA), раздел `1.5 Пр Декомпозиция/Dataset/`.

Поместить файл в папку `data/` в корне проекта. Если папка отсутствует — создать её.

Файлы из `data/` не отслеживаются Git и в репозиторий не загружаются.

**4. Загрузка данных в Python**

```python
from src.data_loader import load_incidents

df = load_incidents("data/iavd_incidents_raw_target.parquet")
```

Функция возвращает DataFrame с 240 295 строками и 100 столбцами (97 исходных и 3 дополнительных).

Результаты исследования данных и описание ограничений приведены в [отчёте](docs/data_report.md).