"""Дашборд Taipy «Карточки SOC» — задание 1.

Запуск из корня репозитория:
    poetry run python task_1/app.py
    poetry run python task_1/app.py --data data/iavd_1000_incidents_raw_target.parquet
и откройте http://127.0.0.1:5011
"""

from __future__ import annotations

import argparse
from pathlib import Path

from taipy.gui import Gui

import loader
from charts import PLOT_CONFIG  # noqa: F401  (используется в разметке страницы)
from loader import DAY_FILTERS, PERIODS, SEVERITY_ORDER, Filters
from views import TILES, compute

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Дашборд «Карточки SOC» (задание 1)")
parser.add_argument("--data", type=Path, default=None, help="parquet с карточками (по умолчанию — полный файл из data/, если он есть, иначе выборка)")
parser.add_argument("--port", type=int, default=5011)
args, _ = parser.parse_known_args()

DATA = loader.load(args.data)

PERIOD_LOV = list(PERIODS)
DAY_LOV = list(DAY_FILTERS)
SEVERITY_LOV = list(SEVERITY_ORDER)

# ---------------------------------------------------------------- состояние по умолчанию
period = PERIOD_LOV[0]
days = DAY_LOV[0]
severities = list(SEVERITY_LOV)
severity_summary = "Все уровни"
globals().update(compute(DATA, Filters()))


def summary_of(chosen) -> str:
    if not chosen or set(chosen) == set(SEVERITY_LOV):
        return "Все уровни"
    return ", ".join(level for level in SEVERITY_LOV if level in chosen)


def filters_of(state) -> Filters:
    chosen = tuple(state.severities or ())
    return Filters(state.period, state.days, () if set(chosen) == set(SEVERITY_LOV) else chosen)


def on_filter(state, var_name=None, value=None):
    values = compute(DATA, filters_of(state))
    values["severity_summary"] = summary_of(state.severities)
    with state:
        for name, item in values.items():
            setattr(state, name, item)


def on_reset(state, *_):
    with state:
        state.period = PERIOD_LOV[0]
        state.days = DAY_LOV[0]
        state.severities = list(SEVERITY_LOV)
    on_filter(state)


# ---------------------------------------------------------------- разметка
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
<|Карточки SOC: что приходит, что из этого шум и где SOC не успевает|text|class_name=title|>
<|{{caption}}|text|class_name=caption|>
|>

<|layout|columns=200px 240px 240px 1fr|gap=12px|class_name=filters|
<|{{period}}|selector|lov={{PERIOD_LOV}}|dropdown|label=Период|on_change=on_filter|>

<|{{days}}|selector|lov={{DAY_LOV}}|dropdown|label=Тип дня|on_change=on_filter|>

<|{{severities}}|selector|lov={{SEVERITY_LOV}}|dropdown|multiple|label=Критичность|selection_message={{severity_summary}}|on_change=on_filter|>

<|Сбросить фильтры|button|on_action=on_reset|class_name=reset|>
|>

{kpi_markup()}

<|layout|columns=1 1|gap=16px|class_name=row|
{card("1. Из чего состоит поток карточек и как он менялся?", "Карточки по вердикту аналитика, {flow_note}", "fig_flow", 330)}
{card("2. Какие правила корреляции создают больше всего шума?", "10 правил с наибольшим числом карточек; справа — карточек и доля потока", "fig_rules", 330)}
|>

<|layout|columns=1 1|gap=16px|class_name=row|
{card("3. Берут ли критические карточки в работу быстрее остальных?", "Накопленная доля карточек с корректным временем реакции", "fig_react", 330)}
{card("4. Когда SOC не укладывается в SLA взятия в работу?", "Доля карточек, взятых позже норматива SLA; пустая ячейка — меньше 5 карточек", "fig_sla", 330)}
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
    gui.run(title="Карточки SOC", host="127.0.0.1", port=args.port, dark_mode=False, stylekit=STYLEKIT,
            watermark="", use_reloader=False, run_browser=False)
