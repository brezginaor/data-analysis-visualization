"""Собирает task_2/REPORT.md: ответы на пять вопросов с числами из тех же расчётов, что и плитки.

Запуск из корня репозитория:
    poetry run python task_2/report.py
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np

import loader
from loader import TRAIN_MONTHS, Filters, month_label, ru_int, ru_pct
from views import summary

HERE = Path(__file__).resolve().parent


def build(predictions: Path | None, dataset: Path | None) -> str:
    data = loader.load(predictions, dataset)
    s = summary(data, Filters())
    c, monthly, cats, risk = s["confusion"], s["monthly"], s["categories"], s["risk"]
    qx, qy = s["gains"]["threshold"]
    every = int(round(1 / c["precision"])) if c["precision"] else None
    systematic = cats.loc[cats["low"] > c["miss"]]
    low_rows = risk.loc[risk["row"].str.startswith("Низкий")]
    low_real = int(low_rows["real"].sum())
    raised_low = risk.iloc[1]
    critical = risk.iloc[-1]
    months_text = "; ".join(
        f"{month_label(r.month).lower()} — {ru_pct(r.miss)} ({ru_int(r.FN)} из {ru_int(r.incidents)})"
        for r in monthly.itertuples())

    lines = [
        "# Можно ли доверять модели-советнику SOC: отчёт к дашборду (задание 2)",
        "",
        f"Модель `wlarf-dec-jun-v3`, порог 0,18: оценка ≥ 0,18 — «инцидент», карточка идёт в приоритет. "
        f"Лог модели: {month_label(data.predictions['month'].min()).lower()} — "
        f"{month_label(data.predictions['month'].max()).lower()}, {ru_int(c['n'])} карточек с вердиктом "
        f"(«Решён» — реальный инцидент, «False Positive» — ложное срабатывание); ещё {ru_int(data.excluded)} "
        f"оценок без такого вердикта в метрики не входят. Собран `report.py` {date.today():%d.%m.%Y}.",
        "",
        "## Коротко для руководителя SOC",
        "",
        f"- **Модель пропускает каждый {int(round(1 / c['miss']))}-й реальный инцидент:** "
        f"{ru_int(c['FN'])} из {ru_int(c['incidents'])} ({ru_pct(c['miss'])}, 95 % интервал "
        f"{ru_pct(c['miss_low'])}–{ru_pct(c['miss_high'])}) ушли в отложенные.",
        f"- **Очередь она разгружает сильно:** откладывает {ru_pct(c['deferred'])} карточек. "
        f"Но из поднятых ею карточек инцидентом оказывается только {ru_pct(c['precision'])}"
        + (f" — каждая {every}-я." if every else "."),
        "- **Ошибки систематические:** в категориях "
        + ", ".join(f"«{name}» ({ru_int(r.missed)} из {ru_int(r.incidents)})" for name, r in systematic.iterrows())
        + " модель пропускает инциденты намного чаще среднего." if len(systematic) else
        "- Ни одна категория не выделяется по пропускам сильнее случайного разброса.",
        f"- **Уровню риска «низкий» верить нельзя:** в нём {ru_pct(low_real / c['incidents'])} всех реальных "
        f"инцидентов, в том числе все пропуски. Порог 0,18 лежит внутри «низкого» уровня (до 0,37), "
        f"поэтому {ru_int(raised_low['cards'])} карточек с «низким» риском модель сама же подняла в приоритет.",
    ]
    if s["compare"]:
        train, work = s["compare"]["train"], s["compare"]["work"]
        lines.append(
            f"- **Поток изменился:** доля реальных инцидентов {ru_pct(train['real_share'])} в предполагаемом окне "
            f"обучения и {ru_pct(work['real_share'])} в период работы модели; «Вендор из сием» заполнен у "
            f"{ru_pct(train['vendor_share'], 0)} и {ru_pct(work['vendor_share'], 0)} карточек.")
    lines += [
        "- **Вывод:** как сортировке очереди модели можно доверять — верх очереди она собирает хорошо. "
        "Как фильтру, который откладывает карточки без проверки, — нельзя.",
        "",
        "## 1. Сколько реальных инцидентов модель пропускает и меняется ли это со временем?",
        "",
        "*Точки по неделям с 95 % интервалом и линией среднего.*",
        "",
        f"- Всего пропущено {ru_int(c['FN'])} из {ru_int(c['incidents'])} реальных инцидентов — **{ru_pct(c['miss'])}**.",
        f"- По месяцам: {months_text}.",
        f"- Интервалы недель широкие: в неделе от {ru_int(s['weekly']['incidents'].min())} до "
        f"{ru_int(s['weekly']['incidents'].max())} инцидентов, поэтому скачок одной недели ещё не поломка, "
        "а тренд нужно читать по нескольким неделям подряд.",
        "",
        "## 2. Что даёт SOC разбор очереди в порядке модели?",
        "",
        "*Кривая поиска (cumulative gains) с отметкой рабочего порога.*",
        "",
        f"- На пороге 0,18 модель поднимает **{ru_pct(qx, 0)}** очереди и находит в них **{ru_pct(qy, 0)}** "
        f"реальных инцидентов; без модели в той же доле очереди было бы {ru_pct(qx, 0)} инцидентов.",
        f"- Чтобы найти 90 % инцидентов, по оценке модели нужно разобрать {ru_pct(s['queue_for_90'], 0)} очереди.",
        "",
        "## 3. Где модель «слепа»?",
        "",
        "*Точечная диаграмма: доля пропусков по категориям IRP с 95 % интервалами.*",
        "",
    ]
    for name, r in cats.iterrows():
        mark = " — **систематически выше среднего**" if r.low > c["miss"] else ""
        lines.append(f"- «{name}»: пропущено {ru_int(r.missed)} из {ru_int(r.incidents)} ({ru_pct(r.share, 0)}, "
                     f"интервал {ru_pct(r.low, 0)}–{ru_pct(r.high, 0)}){mark}.")
    lines += [
        "",
        "Систематическая зона — та, где весь интервал лежит правее средней доли пропусков: "
        "там ошибки не объясняются случайностью.",
        "",
        "## 4. Можно ли верить уровню риска модели?",
        "",
        "*Линейчатая диаграмма: доля реальных инцидентов на каждом уровне риска.*",
        "",
    ]
    for r in risk.itertuples():
        lines.append(f"- {r.row}: инцидентом оказались {ru_pct(r.share, 0) if np.isfinite(r.share) else '—'} "
                     f"карточек ({ru_int(r.real)} из {ru_int(r.cards)}; {ru_pct(r.of_all_real, 0)} всех инцидентов).")
    lines += [
        "",
        f"Высоким уровням риска верить можно: на «критическом» инцидентом оказываются {ru_pct(critical['share'], 0)} "
        "карточек. «Низкий» уровень смешивает отложенные карточки и карточки, которые модель подняла в приоритет: "
        "шкалу риска и порог решения нужно согласовать, иначе аналитик видит «низкий риск» у карточки из приоритета.",
        "",
        "## 5. Изменился ли поток карточек с тех пор, как модель обучали?",
        "",
        "*Линии по месяцам основного датасета курса: доля реальных инцидентов и заполненность «Вендора».*",
        "",
    ]
    if s["compare"]:
        train, work = s["compare"]["train"], s["compare"]["work"]
        lines += [
            f"- Окно обучения неизвестно; по названию версии `wlarf-dec-jun-v3` считаем им "
            f"{month_label(TRAIN_MONTHS[0]).lower()} — {month_label(TRAIN_MONTHS[1]).lower()}.",
            f"- Доля реальных инцидентов среди карточек с вердиктом: {ru_pct(train['real_share'])} тогда и "
            f"{ru_pct(work['real_share'])} в период работы модели.",
            f"- «Вендор из сием» (обогащение карточки) заполнен у {ru_pct(train['vendor_share'], 0)} карточек тогда и "
            f"у {ru_pct(work['vendor_share'], 0)} в период работы; по месяцам работы — всё реже.",
            "- Модель работает на потоке, который заметно отличается от того, на котором её, вероятно, учили: "
            "это довод за переобучение на свежих месяцах.",
        ]
    else:
        lines.append("- Нет `iavd_ml_dataset.parquet` в `data/`: сравнение с периодом обучения не построено.")
    lines += [
        "",
        "## Что предлагаем",
        "",
        "1. Использовать модель для порядка разбора, а не как фильтр: отложенные карточки всё равно просматривать.",
        "2. Карточки из категорий-«слепых зон» разбирать без учёта оценки модели.",
        "3. Согласовать шкалу риска с порогом решения: «низкий» не должен включать карточки из приоритета.",
        "4. Переобучить модель на свежих месяцах и вернуть заполнение обогащения («Вендор из сием»).",
        "",
        "## Оговорки",
        "",
        "- Метрики считаются только по вердиктам «Решён» и «False Positive»; спам, дубликаты и оценки без карточки "
        "не входят.",
        "- Пропуск — реальный инцидент с оценкой ниже 0,18. В жизни его всё равно разобрал аналитик: это то, что "
        "SOC потерял бы, если бы доверял модели без проверки.",
        "- 95 % интервалы — интервалы Уилсона для доли.",
        "- Даты и недели — по Москве.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "REPORT.md")
    args = parser.parse_args()
    args.out.write_text(build(args.predictions, args.dataset), encoding="utf-8")
    print(f"Готово: {args.out}")
