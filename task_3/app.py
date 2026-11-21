"""Дашборд SHAP-объяснений для команды № 4.

Запуск: poetry run python task_3/app.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from taipy.gui import Gui

import loader
from charts import PLOT_CONFIG  # noqa: F401  (используется в разметке страницы)
from loader import OUTCOMES, PROJECT_TEAM, Filters
from views import TILES, card_labels, compute, entity_of

parser = argparse.ArgumentParser(description="Дашборд «Почему модель решила так» (задание 3, SHAP)")
parser.add_argument("--cards", type=Path, default=None, help="Путь к CSV с карточками")
parser.add_argument("--shap", type=Path, default=None, help="Путь к CSV с SHAP-вкладами")
parser.add_argument("--port", type=int, default=5014)
args, _ = parser.parse_known_args()

DATA = loader.load(args.cards, args.shap)
print(f"Команда SHAP № {PROJECT_TEAM}: {len(DATA.cards)} карточек; данные: {DATA.path.parent}")

OUTCOME_LOV = list(OUTCOMES)
selected_outcomes = list(OUTCOME_LOV)
outcome_summary = "Все исходы"
cards = card_labels(DATA, Filters())
selected_card = cards[0]
globals().update(compute(DATA, Filters(), entity_of(selected_card)))


def filters_of(state) -> Filters:
    outcomes = tuple(state.selected_outcomes or ())
    return Filters(outcomes=() if set(outcomes) == set(OUTCOME_LOV) else outcomes)


def on_filter(state, var_name=None, value=None):
    f = filters_of(state)
    if var_name == "selected_outcomes":
        state.cards = card_labels(DATA, f)
        state.selected_card = state.cards[0] if state.cards else ""
    entity = entity_of(state.selected_card) if state.selected_card else None
    values = compute(DATA, f, entity)
    values["outcome_summary"] = summary_of(state.selected_outcomes)
    with state:
        for name, item in values.items():
            setattr(state, name, item)


def summary_of(chosen) -> str:
    if not chosen or set(chosen) == set(OUTCOME_LOV):
        return "Все исходы"
    return ", ".join(chosen)


def on_reset(state, *_):
    with state:
        state.selected_outcomes = list(OUTCOME_LOV)
        state.cards = card_labels(DATA, Filters())
        state.selected_card = state.cards[0]
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
    hint_markup = f"<|{hint}|text|class_name=card-hint|>\n" if hint else ""
    return (
        "<|part|class_name=card|\n"
        f"<|{question}|text|class_name=card-title|>\n"
        f"{hint_markup}"
        f"<|chart|figure={{{figure}}}|plot_config={{PLOT_CONFIG}}|height={height}px|>\n"
        "|>"
    )


page = f"""
<|part|class_name=shell|
<|part|class_name=header|
<|Почему модель решила так: SHAP-объяснение для SOC|text|class_name=title|>
<|{{caption}}|text|class_name=caption|>
|>

<|layout|columns=280px 1fr 180px|gap=12px|class_name=filters|
<|{{selected_outcomes}}|selector|lov={{OUTCOME_LOV}}|dropdown|multiple|label=Исходы|selection_message={{outcome_summary}}|on_change=on_filter|>

<|{{selected_card}}|selector|lov={{cards}}|dropdown|label=Карточка (водопад)|on_change=on_filter|>

<|Сбросить фильтры|button|on_action=on_reset|class_name=reset|>
|>

{kpi_markup()}

<|layout|columns=1 1|gap=16px|class_name=row|
{card("1. Почему модель решила так по этой карточке?",
      "Вклады складываются в логарифмах шансов: вправо — к инциденту, влево — к легитимному",
      "fig_card", 430)}
{card("2. Какие признаки дают наибольшие вклады в нашей группе?",
      "Средний модуль SHAP по карточкам выбранных фильтров (не по всему потоку SOC)",
      "fig_flow", 430)}
|>

<|layout|columns=1 1|gap=16px|class_name=row|
{card("3. Куда толкает признак у инцидентов, а куда у ложных тревог?",
      "Положительный SHAP — к инциденту, отрицательный — к легитимному; средние по исходам",
      "fig_outcomes", 360)}
{card("4. Почему модель подняла ложную тревогу?", "{doubt_caption}", "fig_alert", 360)}
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
    gui.run(title="Почему модель решила так (SHAP)", host="127.0.0.1", port=args.port, dark_mode=False,
            stylekit=STYLEKIT, watermark="", use_reloader=False, run_browser=False)
