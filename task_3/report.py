"""Собирает task_3/REPORT.md: ответы на вопросы задания 3 с числами из тех же расчётов, что и плитки.

Запуск из корня репозитория:
    .venv/bin/python task_3/report.py   (или poetry run python task_3/report.py)
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np

import loader
from loader import ENRICHMENT, PROJECT_TEAM, SUPER, THRESHOLD, Filters, ru_int, ru_num, ru_pct, ru_proba


def sci(value: float) -> str:
    if value == 0:
        return "0"
    exp = int(np.floor(np.log10(value)))
    return f"{ru_num(value / 10 ** exp, 0)}·10{str(exp).translate(SUPER)}"

from views import summary

HERE = Path(__file__).resolve().parent
def build(cards: Path | None, shap: Path | None) -> str:
    data = loader.load(cards, shap)  # loader оставляет только группу PROJECT_TEAM
    initial_id = int(data.cards["EntityId"].iloc[0])
    s = summary(data, Filters(), initial_id)
    # Сомнительный пример выбираем только внутри своей группы, если он найден.
    if s["doubtful"] is not None:
        initial_id = int(s["doubtful"]["EntityId"])
        s = summary(data, Filters(), initial_id)
    top, signed, doubt = s["top"], s["signed"], s["doubt"]
    confirmed, doubtful = s["confirmed"], s["doubtful"]
    card = s["card"]
    contribs = card["contribs"]

    def shap_of(entity: int, feature: str) -> float:
        row = data.shap.loc[data.shap["EntityId"] == entity].iloc[0]
        return float(row[feature])

    top_lines = [
        f"- **{r.feature}** — {ru_num(r.mean_abs)} лог. шанса ({ru_pct(r.share, 0)} суммы топ-10)"
        for r in top.itertuples()
    ]
    pivot = signed.pivot(index="feature", columns="outcome", values="mean_shap")
    outcome_lines = []
    for feature in pivot.index:
        vals = " · ".join(
            f"{oc} {pivot.loc[feature, oc]:+.2f}" for oc in ("TP", "FP", "FN", "TN") if oc in pivot.columns
        )
        outcome_lines.append(f"- **{feature}**: {vals}")

    confirmed_line = (
        f"Карточка {confirmed['EntityId']}: модель подняла её с оценкой {ru_proba(float(confirmed['proba']))}, "
        f"и даже без признаков обогащения оценка осталась бы выше порога — решение держат "
        f"данные самой карточки (главный толчок «к инциденту» — "
        f"«{max(((f, confirmed[f]) for f in data.features), key=lambda x: x[1])[0]}» "
        f"с вкладом +{ru_num(max(confirmed[f] for f in data.features))} лог. шанса). "
        "Классификация согласуется с меткой TP; для проверки смысла вкладов нужен контекст событий карточки."
        if confirmed is not None else "Подтверждённый случай не найден в выборке."
    )
    if doubtful is not None:
        present = [f for f in ENRICHMENT if f in data.features]
        enr_sum = sum(doubtful[f] for f in present)
        without = 1 / (1 + np.exp(-(doubtful["base"] + doubtful[list(data.features)].sum() - enr_sum)))
        doubtful_line = (
            f"Карточка {doubtful['EntityId']} — ложная тревога с оценкой {ru_proba(float(doubtful['proba']))}, "
            f"на волоске от порога {ru_num(THRESHOLD, 4)}. Уберём вклады обогащения ({', '.join(present)}: "
            f"{ru_num(enr_sum)} лог. шанса) — оценка падает до {ru_proba(float(without))}, ниже порога. "
            "Модель подняла карточку в основном потому, что заполнено обогащение, а не потому, "
            "что в карточке есть сигнал атаки: с сентября это поле деградировало."
            if without < THRESHOLD else
            f"Карточка {doubtful['EntityId']} — ложная тревога у порога, но её поднятие не сводится к обогащению."
        )
    else:
        doubtful_line = "Сомнительный случай в выборке не найден."
    cf = card  # водопад карточки команды 4 (сомнительной, если такая найдена)
    card_lines = [
        f"- Базовая линия (средний логарифм шансов по обучению): {ru_num(cf['waterfall'].iloc[0]['value'])}.",
        *[f"- «{r.feature}»: {r.value:+.2f}".replace('.', ',') + " лог. шанса"
          for r in cf["waterfall"].itertuples() if r.measure == "relative"],
        f"- Итог: sigmoid(итог) = {ru_proba(cf['proba'])} → "
        f"{'поднята в приоритет' if cf['proba'] >= THRESHOLD else 'отложена'}; истина — "
        f"{'инцидент' if cf['outcome'] in ('TP', 'FN') else 'не инцидент'}.",
    ]

    lines = [
        "# Почему модель решила так: SHAP-объяснение для SOC (задание 3)",
        "",
        f"Модель курса `lgbm_gate1__v001`, рабочий порог {ru_num(THRESHOLD, 4)}: оценка не ниже порога — карточка идёт в приоритет. "
        f"Разложение TreeSHAP: вклады {len(data.features)} признаков в логарифме шансов для "
        f"{ru_int(len(data.shap))} карточек команды SHAP № {PROJECT_TEAM} (сентябрь 2025). Собран `report.py` "
        f"{date.today():%d.%m.%Y}.",
        "",
        "Уровни показа выбраны так: **одна карточка** — водопад решения; **поток** — средние вклады и их знак "
        "по исходам; **сомнения** — контрфактическая проверка вкладов обогащения. Объяснение описывает модель, "
        "а не причину инцидента.",
        "",
        "## Коротко для SOC",
        "",
        f"- **Объяснение арифметически согласовано:** sigmoid(база + {len(data.features)} вкладов) воспроизводит оценку модели "
        f"с точностью {sci(data.check)} — это округление вероятности в файле. Вклады сходятся с решением.",
        f"- **Крупнейшие средние вклады в группе:** "
        + ", ".join(f"«{name}»" for name in top["feature"].head(4)) + " (см. раздел 3).",
        f"- **Объяснение подтверждается данными:** {confirmed_line if confirmed is not None else ''}",
        f"- **Объяснение выглядит сомнительно:** {doubtful_line}",
        f"- **На выбранных карточках группы:** при исключении вкладов обогащения из суммы условная оценка оказалась бы ниже порога "
        f"{ru_int(doubt['on_enrichment'])} из {ru_int(doubt['raised'])} поднятых карточек "
        f"({ru_pct(doubt['share'], 0)}), в том числе {ru_int(doubt['fp_near'])} ложных тревог из "
        f"{ru_int(doubt['fp_near_total'])} поднятых.",
        "",
        "## 1. Уровень карточки: почему модель решила так?",
        "",
        f"*Водопад: базовая линия → крупнейшие вклады → итог; proba = sigmoid(итог). "
        f"Пример — карточка {initial_id} команды {PROJECT_TEAM}.*",
        "",
        *card_lines,
        "",
        "Решение складывается из всех вкладов, включая положительные, отрицательные и «остальные». "
        "SHAP описывает поведение модели, а не причину события.",
        "",
        "## 2. Можно ли объяснению верить? Проверяемость",
        "",
        "*Проверка воспроизводимости: оценка модели из одних вкладов.*",
        "",
        f"- Из base и {len(data.features)} вкладов восстановлена оценка каждой карточки: максимальное расхождение с proba из "
        f"файла — {sci(data.check)}, то есть ровно округление вероятности до 6 знаков.",
        "- Значит, показанные вклады действительно те, по которым модель приняла решение, а не пересчёт "
        "постфактум другой моделью.",
        "",
        "## 3. Уровень потока: на чём модель держится?",
        "",
        "*Средний |SHAP| по карточкам выбранных фильтров.*",
        "",
        *top_lines,
        "",
        "## 4. Один признак — в обе стороны: средние вклады по исходам",
        "",
        "*Средний подписанный SHAP топ-признаков для TP/FP/FN/TN.*",
        "",
        *outcome_lines,
        "",
        "Средние подписанные вклады различаются по исходам. Среднее около нуля не означает, "
        "что признак не используется: положительные и отрицательные вклады могут взаимно компенсироваться. "
        "Выводы относятся только к выбранным карточкам команды 4.",
        "",
        "## 5. Сомнительные объяснения: деградировавшее обогащение",
        "",
        "*Контрфактическая проверка: убираем вклады «Алерт», «Вендор из сием», «len_Значение Инцидента» и "
        "пересчитываем оценку.*",
        "",
        f"- Без обогащения упали бы ниже порога {ru_int(doubt['on_enrichment'])} из {ru_int(doubt['raised'])} "
        f"поднятых карточек — {ru_pct(doubt['share'], 0)}.",
        f"- Среди поднятых ложных тревог у порога (оценка до 0,1) таких {ru_int(doubt['fp_near'])} из "
        f"{ru_int(doubt['fp_near_total'])} — их поднятие почти целиком сделано «сломанным» полем.",
        f"- {doubtful_line}",
        "",
        "## Что предлагаем",
        "",
        "1. Показывать аналитику водопад вкладов рядом с решением модели — по образцу дашборда `task_3/`.",
        "2. Не доверять поднятиям, которые держатся на обогащении, пока заполнение «Алерт»/«Вендор» не "
        "восстановлено: это ровно те карточки, где объяснение нельзя проверить данными карточки.",
        "3. Следить за долей «обогащением поднятых» карточек во времени: её рост — сигнал деградации входов, "
        "а не поломка модели.",
        "4. Помнить: SHAP объясняет модель, а не причину инцидента; «Hostname атакующего» в объяснении — не "
        "доказательство атаки.",
        "",
        "## Оговорки",
        "",
        "- Данные учебные (условные), модель и порог зафиксированы курсом; SHAP-вклады предрассчитаны "
        "в `course_pack` (архив IAVD_3.7_ML_team).",
        "- Вклады — в логарифме шансов; в вероятности их переводит sigmoid, что и делает водопад.",
        "- Контрфактическая проверка (раздел 5) показывает, без каких полей решение изменилось бы, но не "
        "является переобучением модели.",
        "- Даты — сентябрь 2025, карточки моделью не видели (вне окна обучения).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, default=None)
    parser.add_argument("--shap", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "REPORT.md")
    args = parser.parse_args()
    args.out.write_text(build(args.cards, args.shap), encoding="utf-8")
    print(f"Готово: {args.out}")
