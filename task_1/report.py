"""Собирает task_1/REPORT.md: ответы на вопросы дашборда с числами из тех же расчётов, что и плитки.

Запуск из корня репозитория (по умолчанию — полный файл из data/, если он есть):
    poetry run python task_1/report.py
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np

import loader
import metrics
from loader import FP, NO_VERDICT, OTHER, REAL, Filters, ru_duration, ru_int, ru_num, ru_pct
from views import SHORT_DAY, dataset_label, summary

HERE = Path(__file__).resolve().parent


def cards_word(n: int) -> str:
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return "карточка"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return "карточки"
    return "карточек"


def times(a: float, b: float) -> str:
    return f"{ru_num(a / b)} раза" if b else "—"


def section_flow(data, s) -> list[str]:
    flow = s["flow"]
    cards = data.cards
    weekly = cards.groupby(cards["date"] - loader.pd.to_timedelta(cards["date"].dt.dayofweek, unit="D")).size()
    peak_week, peak = weekly.idxmax(), int(weekly.max())
    shares = cards["outcome"].value_counts(normalize=True)
    return [
        "## 1. Из чего состоит поток карточек и как он менялся?",
        "",
        "*Столбчатая диаграмма с накоплением по неделям, цвет — вердикт аналитика.*",
        "",
        f"- За {ru_int(flow['days'])} дн. поступило **{ru_int(flow['cards'])}** карточек — "
        f"≈ {ru_num(flow['per_day'], 0 if flow['per_day'] >= 10 else 1)} в день. "
        f"Самая нагруженная неделя — с {peak_week:%d.%m.%Y}: {ru_int(peak)} {cards_word(peak)}.",
        f"- **{ru_pct(shares.get(FP, 0))}** карточек закрыты как False Positive, реальных инцидентов — "
        f"**{ru_int(flow['real'])}** ({ru_pct(shares.get(REAL, 0))}), дубликаты и прочие — "
        f"{ru_pct(shares.get(OTHER, 0))}, без вердикта — **{ru_pct(shares.get(NO_VERDICT, 0))}**.",
        "- Карточки без вердикта занижают и долю FP, и долю реальных инцидентов: "
        "по ним неизвестно, чем закончился разбор.",
        "",
    ]


def section_rules(data, s) -> list[str]:
    rules, table = s["rules"], s["top_rules"]
    quiet = metrics.quiet_rules(table)
    useful = table.assign(real_share=table[REAL] / table["total"]).sort_values("real_share", ascending=False)
    leader = table.iloc[0]
    lines = [
        "## 2. Какие правила корреляции создают больше всего шума?",
        "",
        "*Линейчатая диаграмма с накоплением: 10 правил с наибольшим числом карточек, цвет — вердикт.*",
        "",
        f"- 10 правил из {ru_int(rules['rules_total'])} дают **{ru_pct(rules['top_share'])}** всех карточек. "
        f"Лидер — `{rules['leader']}`: {ru_pct(rules['leader_share'])} потока, из них "
        f"{ru_pct(leader[NO_VERDICT] / leader['total'])} без вердикта и {ru_pct(leader[FP] / leader['total'])} False Positive.",
    ]
    if len(quiet):
        lines.append(
            f"- **{len(quiet)}** из 10 правил почти не дают реальных инцидентов (меньше 1 % карточек) и создают "
            f"**{ru_pct(rules['quiet_share'])}** потока — первые кандидаты на донастройку, белые списки или автозакрытие: "
            + ", ".join(f"`{name}` ({ru_pct(row[REAL] / row['total'])} реальных)" for name, row in quiet.iterrows()) + "."
        )
    else:
        lines.append("- У каждого правила топа есть реальные инциденты (не меньше 1 % карточек): "
                     "шум нужно снижать донастройкой, а не отключением правил.")
    if len(useful) and useful["real_share"].iloc[0] > 0:
        lines.append(
            f"- Самое «полезное» правило топа — `{useful.index[0]}`: "
            f"{ru_pct(useful['real_share'].iloc[0])} его карточек — реальные инциденты."
        )
    return [*lines, ""]


def section_reaction(data, s) -> list[str]:
    react = s["react"]
    medians = react["medians"]
    crit, rest = react["median_critical"], react["median_rest"]
    lines = [
        "## 3. Берут ли критические карточки в работу быстрее остальных?",
        "",
        "*Кривые накопленной доли (ECDF) по уровням критичности, логарифмическая шкала времени.*",
        "",
    ]
    if np.isfinite(crit) and np.isfinite(rest):
        if crit > rest:
            verdict = (f"**Нет.** Медиана времени до взятия в работу у критических карточек — **{ru_duration(crit)}**, "
                       f"у остальных — {ru_duration(rest)}: критические ждут в {times(crit, rest)} дольше. "
                       "Приоритет в очереди сейчас не ускоряет реакцию.")
        else:
            verdict = (f"**Да.** Медиана у критических карточек — **{ru_duration(crit)}**, "
                       f"у остальных — {ru_duration(rest)}.")
        lines.append(f"- {verdict}")
    lines.append("- Медианы по уровням: " + "; ".join(
        f"{level.lower()} — {ru_duration(value)}" for level, value in medians.items()) + ".")
    lines.append(f"- У {ru_int(react['missing'])} карточек ({ru_pct(react['missing_share'])}) нет даты взятия в работу "
                 "или интервал ошибочный: их закрыли, не взяв в работу, и в кривые они не входят. "
                 f"Кривые построены по {ru_int(react['known'])} карточкам.")
    return [*lines, ""]


def section_sla(data, s) -> list[str]:
    sla = s["sla"]
    share, counts = s["sla_matrix"]
    cards = data.cards
    known = cards.loc[cards["sla_breach"].notna()]
    by_day = known.groupby(known["day_type"].isin(loader.WORKDAY_TYPES))["sla_breach"].mean()
    lines = [
        "## 4. Когда SOC не укладывается в SLA взятия в работу?",
        "",
        "*Тепловая карта: тип дня (по производственному календарю) × трёхчасовое окно суток, "
        "цвет — доля нарушений.*",
        "",
        f"- Норматив взятия в работу нарушен у **{ru_pct(sla['breach_share'])}** карточек "
        f"({ru_int(sla['breaches'])} из {ru_int(sla['known'])} с известной датой взятия).",
    ]
    if sla["worst_cell"] is not None:
        row, col = sla["worst_cell"]
        lines.append(f"- Хуже всего — {SHORT_DAY.get(row, row)}, {col} ч по Москве: **{ru_pct(sla['worst_share'], 0)}** "
                     f"карточек взяты позже норматива ({ru_int(sla['worst_cards'])} {cards_word(sla['worst_cards'])} "
                     f"в окне); лучшее окно — {ru_pct(sla['best_share'], 0)}. Худшее и лучшее окна выбираются "
                     f"среди ячеек, где не меньше {ru_int(sla['headline_min'])} карточек: в остальных доля случайна.")
    if True in by_day and False in by_day:
        lines.append(f"- Рабочие дни — {ru_pct(float(by_day[True]))} нарушений, "
                     f"выходные и праздники — {ru_pct(float(by_day[False]))}.")
    return [*lines, ""]


def build(path: Path | None) -> str:
    data = loader.load(path)
    s = summary(data, Filters())
    sample = data.path.name != loader.FULL_FILE.name
    head = [
        "# Карточки SOC: отчёт к дашборду (задание 1)",
        "",
        f"Данные: `{data.path.name}` ({dataset_label(data)}). "
        f"Фильтры по умолчанию: весь период, все дни, все уровни критичности. "
        f"Собран `report.py` {date.today():%d.%m.%Y}.",
        "",
    ]
    if sample:
        head += [
            "> **Отчёт собран на учебной выборке из 1 000 карточек — выводы по ней ненадёжны.** "
            "Для сдачи положите `iavd_incidents_raw_target.parquet` в `data/` и выполните "
            "`poetry run python task_1/report.py`.",
            "",
        ]
    body = section_flow(data, s) + section_rules(data, s) + section_reaction(data, s) + section_sla(data, s)
    tail = [
        "## Оговорки",
        "",
        "- Время до взятия в работу = `Дата взятия в работу` − `Создан`. Поля `Время решения` и `Время Решения` "
        "не используются: их единицы неоднородны (см. `docs/data_report.md`).",
        "- Отрицательные интервалы (взятие раньше создания) считаются ошибкой данных и исключаются, а не обнуляются.",
        "- Нарушение SLA = интервал больше `SLA по взятию в работу Атаки` (норматив в миллисекундах). "
        "Карточки без даты взятия или без норматива в эту долю не входят.",
        "- Время суток и даты — по Москве; тип дня — из `calendar.parquet`, для дат вне календаря — по дню недели.",
        "- «Реальный инцидент» — вердикт «Решён», «Без вердикта» — «Не указан», "
        "«Дубликат и прочие» — остальные вердикты.",
        "",
    ]
    return "\n".join(head + body + tail)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "REPORT.md")
    args = parser.parse_args()
    text = build(args.data)
    args.out.write_text(text, encoding="utf-8")
    print(f"Готово: {args.out}")
