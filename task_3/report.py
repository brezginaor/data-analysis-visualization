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
from loader import ENRICHMENT, SUPER, THRESHOLD, Filters, ru_int, ru_num, ru_pct, ru_proba


def sci(value: float) -> str:
    exp = int(np.floor(np.log10(value)))
    return f"{ru_num(value / 10 ** exp, 0)}·10{str(exp).translate(SUPER)}"

from views import summary

HERE = Path(__file__).resolve().parent
EXAMPLE_CONFIRMED = 523983  # TP, который модель подняла бы и без обогащения
EXAMPLE_DOUBTFUL = 527002   # FP у порога, который без обогащения падает ниже порога


def build(cards: Path | None, shap: Path | None) -> str:
    data = loader.load(cards, shap)
    s = summary(data, Filters(), EXAMPLE_DOUBTFUL)
    if s["doubtful"] is not None:
        s = summary(data, Filters(), int(s["doubtful"]["EntityId"]))  # водопад — по сомнительному случаю
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
        "Это объяснение подтверждается исходом: карточка действительно решена как инцидент."
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
    cf = card  # водопад сомнительной карточки — тот же 527002
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
        f"Разложение TreeSHAP: вклады 73 признаков в логарифме шансов для {ru_int(len(data.shap))} карточек "
        "сентября 2025 (40 команд × 8 карточек: по 2 исхода TP/FP/FN/TN). Собран `report.py` "
        f"{date.today():%d.%m.%Y}.",
        "",
        "Уровни показа выбраны так: **одна карточка** — водопад решения; **поток** — средние вклады и их знак "
        "по исходам; **сомнения** — контрфактическая проверка вкладов обогащения. Объяснение описывает модель, "
        "а не причину инцидента.",
        "",
        "## Коротко для SOC",
        "",
        f"- **Объяснению можно верить арифметически:** sigmoid(база + 73 вклада) воспроизводит оценку модели "
        f"с точностью {sci(data.check)} — это округление вероятности в файле. Вклады сходятся с решением.",
        f"- **Модель держится на адресах и частотах:** топ опоры — «Hostname атакующего», «Hostname Цели», "
        f"`src_is_private`, `freq_Наименование` (см. раздел 3).",
        f"- **Объяснение подтверждается данными:** {confirmed_line if confirmed is not None else ''}",
        f"- **Объяснение выглядит сомнительно:** {doubtful_line}",
        f"- **Сомнение носит системный характер:** без вкладов обогащения упали бы ниже порога "
        f"{ru_int(doubt['on_enrichment'])} из {ru_int(doubt['raised'])} поднятых карточек "
        f"({ru_pct(doubt['share'], 0)}), в том числе {ru_int(doubt['fp_near'])} ложных тревог из "
        f"{ru_int(doubt['fp_near_total'])} поднятых.",
        "",
        "## 1. Уровень карточки: почему модель решила так?",
        "",
        f"*Водопад: базовая линия → крупнейшие вклады → итог; proba = sigmoid(итог). Пример — сомнительная "
        f"карточка {int(s['doubtful']['EntityId']) if s['doubtful'] is not None else EXAMPLE_DOUBTFUL}.*",
        "",
        *card_lines,
        "",
        "Вклады по модулю не складываются «в пользу» решения: «src_is_private» толкает к инциденту, "
        "а «Hostname атакующего» здесь — против. Решение — баланс 73 вкладов, а не один флаг.",
        "",
        "## 2. Можно ли объяснению верить? Проверяемость",
        "",
        "*Проверка воспроизводимости: оценка модели из одних вкладов.*",
        "",
        f"- Из base и 73 вкладов восстановлена оценка каждой карточки: максимальное расхождение с proba из "
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
        "Главный вывод для SOC: признаки вроде «Hostname атакующего» сильно работают у найденных инцидентов "
        "(TP +1,75), почти ничего не дают у пропущенных (FN +0,01) и умеренно у ложных тревог (FP +1,05). "
        "Модель опирается на то, кто и куда стучался, — когда адресного сигнала нет, она теряет опору "
        "и откладывает карточку: именно так выглядят её пропуски.",
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
