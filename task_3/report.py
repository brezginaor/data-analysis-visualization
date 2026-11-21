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
        f"а при арифметическом исключении выбранных SHAP-вкладов условная оценка остаётся "
        f"выше порога. Наибольший положительный вклад даёт "
        f"«{max(((f, confirmed[f]) for f in data.features), key=lambda x: x[1])[0]}» "
        f"(+{ru_num(max(confirmed[f] for f in data.features))} лог. шанса). "
        "Исход TP подтверждает правильность классификации этой карточки, но не смысл "
        "SHAP-объяснения: исходные значения признаков и контекст события в этом анализе не проверялись."
        if confirmed is not None else "Подтверждённый случай не найден в выборке."
    )
    if doubtful is not None:
        present = [f for f in ENRICHMENT if f in data.features]
        enr_sum = sum(doubtful[f] for f in present)
        without = 1 / (1 + np.exp(-(doubtful["base"] + doubtful[list(data.features)].sum() - enr_sum)))
        doubtful_line = (
            f"Карточка {doubtful['EntityId']} — ложная тревога с оценкой {ru_proba(float(doubtful['proba']))}, "
            f"при рабочем пороге {ru_num(THRESHOLD, 4)}. Если арифметически исключить "
            f"SHAP-вклады выбранных признаков ({', '.join(present)}: "
            f"сумма {ru_num(enr_sum)} лог. шанса), условная оценка составит "
            f"{ru_proba(float(without))} — ниже порога. Эти вклады требуют проверки "
            "по исходным данным; это не новый прогноз модели и не доказательство "
            "ухудшения качества обогащения."
            if without < THRESHOLD else
            f"Карточка {doubtful['EntityId']} — ложная тревога; условное исключение выбранных "
            "SHAP-вкладов не меняет сторону порога."
        )
    else:
        doubtful_line = "Сомнительный случай в выборке не найден."
    cf = card  # водопад карточки команды 4 (сомнительной, если такая найдена)
    card_lines = [
        f"- Базовое значение SHAP (логарифм шансов): {ru_num(cf['waterfall'].iloc[0]['value'])}.",
        *[f"- «{r.feature}»: {r.value:+.2f}".replace('.', ',') + " лог. шанса"
          for r in cf["waterfall"].itertuples() if r.measure == "relative"],
        f"- Итог: sigmoid(итог) = {ru_proba(cf['proba'])} → "
        f"{'поднята в приоритет' if cf['proba'] >= THRESHOLD else 'отложена'}; истина — "
        f"{'инцидент' if cf['outcome'] in ('TP', 'FN') else 'не инцидент'}.",
    ]

    if data.raw_check is None:
        raw_audit = (
            "- **Проверка аддитивности в логарифмах шансов:** отдельный `raw_score` "
            "в предоставленных CSV отсутствует. Независимую сверку "
            "`base + ΣSHAP` с `raw_score` при допуске 10⁻⁶ провести нельзя."
        )
    else:
        raw_audit = (
            "- **Проверка аддитивности в логарифмах шансов:** "
            f"максимальное |base + ΣSHAP − raw_score| = {sci(data.raw_check)} "
            "(допуск 10⁻⁶)."
        )

    lines = [
        "# Почему модель решила так: SHAP-объяснение для SOC (задание 3)",
        "",
        f"Модель курса `lgbm_gate1__v001`, рабочий порог {ru_num(THRESHOLD, 4)}: оценка не ниже порога — карточка идёт в приоритет. "
        f"Разложение TreeSHAP: вклады {len(data.features)} признаков в логарифме шансов для "
        f"{ru_int(len(data.shap))} карточек команды SHAP № {PROJECT_TEAM} (сентябрь 2025). Собран `report.py` "
        f"{date.today():%d.%m.%Y}.",
        "",
        f"Уровни показа выбраны так: **одна карточка** — водопад решения; **группа из {ru_int(len(data.shap))} карточек** — средние вклады и их знак "
        "по исходам; **сомнения** — условный пересчёт выбранных SHAP-вкладов. Объяснение описывает модель, "
        "а не причину инцидента.",
        "",
        "## Коротко для SOC",
        "",
        f"- **Восстановление вероятности:** sigmoid(база + {len(data.features)} вкладов) "
        f"согласуется с округлённой `proba` из файла: максимальная разница {sci(data.check)}.",
        raw_audit,
        f"- **Крупнейшие средние вклады в группе:** "
        + ", ".join(f"«{name}»" for name in top["feature"].head(4)) + " (см. раздел 3).",
        f"- **Пример корректно классифицированной тревоги (TP):** {confirmed_line}",
        f"- **Объяснение выглядит сомнительно:** {doubtful_line}",
        f"- **На выбранных карточках группы:** после арифметического исключения выбранных "
        f"SHAP-вкладов условная оценка оказалась бы ниже порога у "
        f"{ru_int(doubt['on_enrichment'])} из {ru_int(doubt['raised'])} поднятых карточек "
        f"({ru_pct(doubt['share'], 0)}). Это не результат повторного запуска модели.",
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
        "*Раздельно проверяем аддитивность в логарифмах шансов и согласованность вероятности.*",
        "",
        raw_audit,
        f"- Восстановление вероятности из `base + ΣSHAP`: максимальное расхождение "
        f"с округлённой `proba` — {sci(data.check)} (допуск 6·10⁻⁷).",
        "- Совпадение округлённой вероятности подтверждает согласованность чисел в файлах, "
        "но не заменяет независимую проверку исходного выхода модели.",
        "",
        "## 3. Уровень группы: какие вклады крупнейшие?",
        "",
        f"*Средний |SHAP| по {ru_int(len(data.shap))} карточкам команды № {PROJECT_TEAM}.*",
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
        "## 5. Сомнительные объяснения: чувствительность к выбранным вкладам",
        "",
        "*Арифметически исключаем SHAP-вклады «Алерт», «Вендор из сием» и "
        "«len_Значение Инцидента (группировка)». Это условный сценарий, не ablation.*",
        "",
        f"- При условном пересчёте оценка стала ниже порога у {ru_int(doubt['on_enrichment'])} "
        f"из {ru_int(doubt['raised'])} поднятых карточек — {ru_pct(doubt['share'], 0)}.",
        f"- Среди {ru_int(doubt['fp_near_total'])} FP с исходной оценкой ниже 10% "
        f"условная оценка опустилась ниже порога у {ru_int(doubt['fp_near'])}. "
        "По этим данным нельзя установить, что признаки обогащения неисправны.",
        f"- {doubtful_line}",
        "",
        "## Что предлагаем",
        "",
        "1. Показывать аналитику водопад вкладов рядом с решением модели — по образцу дашборда `task_3/`.",
        "2. Для сомнительных FP проверить значения «Алерт», «Вендор из сием» и связанных полей "
        "в исходных карточках, прежде чем делать вывод о качестве данных.",
        "3. Если требуется узнать эффект фактического удаления признаков, выполнить отдельный "
        "прогон модели (ablation); по восьми карточкам нельзя доказать деградацию во времени.",
        "4. Помнить: SHAP объясняет модель, а не причину инцидента; «Hostname атакующего» в объяснении — не "
        "доказательство атаки.",
        "",
        "## Оговорки",
        "",
        "- Данные учебные (условные), модель и порог зафиксированы курсом; SHAP-вклады предрассчитаны "
        "в `course_pack` (архив IAVD_3.7_ML_team).",
        "- Вклады — в логарифме шансов; в вероятности их переводит sigmoid, что и делает водопад.",
        "- Условный пересчёт (раздел 5) исключает вклады из уже готового SHAP-разложения, но не "
        "проверяет прогноз модели без этих признаков.",
        "- В выборке только 8 карточек команды 4 за сентябрь 2025; временной динамики качества "
        "обогащения она не показывает.",
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
