"""Дашборд Taipy «Можно ли доверять модели» — задание 2.

Запуск из корня репозитория:
    poetry run python task_2/app.py
и откройте http://127.0.0.1:5013
"""

from __future__ import annotations

import argparse
from pathlib import Path

from taipy.gui import Gui

import loader
from charts import PLOT_CONFIG  # noqa: F401  (используется в разметке страницы)
from loader import SEVERITY_ORDER, Filters, month_label
from views import TILES, compute

parser = argparse.ArgumentParser(description="Дашборд «Можно ли доверять модели» (задание 2)")
parser.add_argument("--predictions", type=Path, default=None, help="лог модели (MODEL_PREDICTIONS_target.parquet)")
parser.add_argument("--dataset", type=Path, default=None, help="основной датасет курса (iavd_ml_dataset.parquet)")
parser.add_argument("--port", type=int, default=5013)
args, _ = parser.parse_known_args()

DATA = loader.load(args.predictions, args.dataset)

MONTHS = {"Весь период": "Весь период"} | {
    month_label(m): m for m in sorted(DATA.predictions["month"].unique())}
PERIOD_LOV = list(MONTHS)
SEVERITY_LOV = [s for s in SEVERITY_ORDER if s in set(DATA.predictions["severity"])]

period = PERIOD_LOV[0]
severities = list(SEVERITY_LOV)
severity_summary = "Все уровни"
globals().update(compute(DATA, Filters()))


def summary_of(chosen) -> str:
    if not chosen or set(chosen) == set(SEVERITY_LOV):
        return "Все уровни"
    return ", ".join(level for level in SEVERITY_LOV if level in chosen)


def filters_of(state) -> Filters:
    chosen = tuple(state.severities or ())
    return Filters(MONTHS[state.period], () if set(chosen) == set(SEVERITY_LOV) else chosen)


def on_filter(state, var_name=None, value=None):
    values = compute(DATA, filters_of(state))
    values["severity_summary"] = summary_of(state.severities)
    with state:
        for name, item in values.items():
            setattr(state, name, item)


def on_reset(state, *_):
    with state:
        state.period = PERIOD_LOV[0]
        state.severities = list(SEVERITY_LOV)
    on_filter(state)


def kpi_markup() -> str:
    cells = []
    for key, label in TILES.items():
        cells.append(
            f"<|part|class_name={{{key}_c}}|\n"
            f"<|{label}|text|class_name=kpi-label|>\n"
            f"<|{{{key}_v}}|text|class_name=kpi-value|>\n"
            f"<|{{{key}_n}}|text|class_name=kpi-note|>\n|>"
        )
    return f"<|layout|columns={' '.join(['1'] * len(cells))}|gap=12px|class_name=kpis|\n" + "\n".join(cells) + "\n|>"


def card(question: str, hint: str, figure: str, height: int) -> str:
    return (
        "<|part|class_name=card|\n"
        f"<|{question}|text|class_name=card-title|>\n"
        f"<|{hint}|text|class_name=card-hint|>\n"
        f"<|chart|figure={{{figure}}}|plot_config={{PLOT_CONFIG}}|height={height}px|>\n"
        "|>"
    )


page = f"""
<|part|class_name=shell|
<|part|class_name=header|
<|Можно ли доверять модели-советнику SOC?|text|class_name=title|>
<|{{caption}}|text|class_name=caption|>
|>

<|layout|columns=220px 240px 1fr|gap=12px|class_name=filters|
<|{{period}}|selector|lov={{PERIOD_LOV}}|dropdown|label=Период работы модели|on_change=on_filter|>

<|{{severities}}|selector|lov={{SEVERITY_LOV}}|dropdown|multiple|label=Критичность|selection_message={{severity_summary}}|on_change=on_filter|>

<|Сбросить фильтры|button|on_action=on_reset|class_name=reset|>
|>

{kpi_markup()}

<|layout|columns=1 1|gap=16px|class_name=row|
{card("1. Сколько реальных инцидентов модель пропускает и меняется ли это со временем?",
      "Точка — неделя, черта — 95 % интервал; серые точки — меньше 10 инцидентов за неделю", "fig_weekly", 330)}
{card("2. Что даёт SOC разбор очереди в порядке модели?",
      "Какая доля реальных инцидентов найдётся, если разбирать карточки от высокой оценки к низкой", "fig_gains", 330)}
|>

<|layout|columns=1 1|gap=16px|class_name=row|
{card("3. Где модель «слепа»?",
      "Доля инцидентов категории, которые модель отложила; черта — 95 % интервал; категории от 10 инцидентов",
      "fig_categories", 330)}
{card("4. Можно ли верить уровню риска модели?",
      "Какая доля карточек каждого уровня риска оказалась реальным инцидентом; справа — инцидентов из карточек",
      "fig_risk", 330)}
|>

<|layout|columns=1|gap=16px|class_name=row|
{card("5. Изменился ли поток карточек с тех пор, как модель обучали?",
      "Основной датасет курса по месяцам; окно обучения — предположение по названию версии модели",
      "fig_history", 320)}
|>
|>
"""

STYLEKIT = {
    "color_primary": "#2a78d6",
    "color_secondary": "#52514e",
    "color_background_light": "#f6f6f3",
    "color_paper_light": "#ffffff",
    "font_family": "system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif",
    "border_radius": 10,
    "root_margin": "0",
}

# Путь к стилям — относительно папки скрипта: так Taipy отдаёт файл браузеру.
gui = Gui(page=page, css_file="styles.css")

if __name__ == "__main__":
    gui.run(title="Можно ли доверять модели", host="127.0.0.1", port=args.port, dark_mode=False,
            stylekit=STYLEKIT, watermark="", use_reloader=False, run_browser=False)
