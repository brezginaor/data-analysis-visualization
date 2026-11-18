"""Проверки расчётов без parquet и без запуска Taipy.

Запуск из корня репозитория:
    poetry run python -m unittest discover -s task_1
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import charts
import loader
import metrics
import views
from loader import FP, NO_VERDICT, OTHER, REAL, Dataset, Filters


def raw_cards() -> pd.DataFrame:
    created = pd.to_datetime([
        "2025-01-06 07:00", "2025-01-06 08:00", "2025-01-07 09:30", "2025-01-10 10:00",  # пн, пн, вт, пт
        "2025-01-11 22:00", "2025-01-12 23:30", "2025-01-13 06:00", "2025-01-13 06:10",  # сб, вс, пн, пн
    ], utc=True)
    return pd.DataFrame({
        "id": range(1, 9),
        "created": created,
        "verdict": ["Решён", "False Positive", "False Positive", "Не указан", "Дубликат", None,
                    "False Positive", "Решен"],
        "severity": ["Критическая", "Средняя", "Средняя", "Высокая", "Низкая", None, "Критическая", "Средняя"],
        "rule": ["R1", "R1", "R1", "R2", "R2~~||~~R3", None, "R1", "R4"],
        "source": ["MaxPatrol SIEM"] * 8,
        # 10 мин, 1 ч, мгновенно, отрицательно, нет даты, 2 ч, 30 с, 5 мин
        "assigned": created + pd.to_timedelta([600, 3600, 0, -60, np.nan, 7200, 30, 300], unit="s"),
        "sla_ms": [900_000, 1_800_000, 600_000, 600_000, 600_000, 3_600_000, 600_000, 0],
    })


def dataset() -> Dataset:
    cards = loader.prepare(raw_cards())
    dates = pd.date_range(cards["date"].min(), cards["date"].max(), freq="D")
    day_type = loader.weekday_day_type(dates)
    cards["day_type"] = cards["date"].map(day_type).astype(int)
    return Dataset(cards=cards, day_type=day_type, path=Path("synthetic.parquet"))


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.cards = dataset().cards

    def test_outcomes_group_verdicts(self):
        self.assertEqual(list(self.cards["outcome"]),
                         [REAL, FP, FP, NO_VERDICT, OTHER, NO_VERDICT, FP, REAL])

    def test_negative_interval_is_missing_not_zero(self):
        self.assertTrue(np.isnan(self.cards.loc[3, "react_sec"]))
        self.assertEqual(self.cards.loc[2, "react_sec"], 0)

    def test_sla_breach_uses_milliseconds_and_skips_unknown(self):
        breach = self.cards["sla_breach"]
        self.assertFalse(breach[0])  # 10 мин < 15 мин
        self.assertTrue(breach[1])  # 1 ч > 30 мин
        self.assertTrue(pd.isna(breach[3]))  # отрицательный интервал
        self.assertTrue(pd.isna(breach[4]))  # нет даты взятия
        self.assertTrue(pd.isna(breach[7]))  # норматив не задан

    def test_multi_value_rule_is_readable_and_missing_filled(self):
        self.assertEqual(self.cards.loc[4, "rule"], "R2 / R3")
        self.assertEqual(self.cards.loc[5, "rule"], "Не указано")
        self.assertEqual(self.cards.loc[5, "severity"], "Отсутствует")

    def test_moscow_time_moves_late_utc_card_to_next_day(self):
        # 2025-01-12 23:30 UTC = 13.01 02:30 МСК
        self.assertEqual(self.cards.loc[5, "date"], pd.Timestamp("2025-01-13"))
        self.assertEqual(self.cards.loc[5, "hour"], 2)

    def test_duplicate_ids_are_rejected(self):
        raw = raw_cards()
        raw.loc[1, "id"] = 1
        with self.assertRaises(ValueError):
            loader.prepare(raw)

    def test_weekday_fallback_matches_calendar_codes(self):
        dates = pd.date_range("2025-01-06", periods=7, freq="D")  # пн..вс
        self.assertEqual(list(loader.weekday_day_type(dates)), [1, 1, 1, 1, 2, 4, 4])


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.data = dataset()

    def test_period_counts_back_from_last_date(self):
        dates = loader.window_dates(self.data, Filters("7 дней"))
        self.assertEqual(dates.max(), self.data.last_date)
        self.assertEqual(len(dates), 7)

    def test_day_type_and_severity_intersect(self):
        selected = loader.apply_filters(self.data, Filters(days="Выходные и праздники"))
        self.assertTrue(selected["day_type"].isin({3, 4}).all())
        selected = loader.apply_filters(self.data, Filters(severities=("Критическая",)))
        self.assertEqual(set(selected["id"]), {1, 7})


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.cards = dataset().cards

    def test_flow_buckets_keep_every_card(self):
        dates = pd.date_range(self.cards["date"].min(), self.cards["date"].max())
        counts, freq = metrics.flow_by_time(self.cards, dates)
        self.assertEqual(freq, "D")
        self.assertEqual(int(counts.to_numpy().sum()), len(self.cards))
        self.assertEqual(len(counts), len(dates))  # пустые дни остаются нулями

    def test_long_window_switches_to_weeks(self):
        dates = pd.date_range("2025-01-01", "2025-06-30")
        _, freq = metrics.flow_by_time(self.cards, dates)
        self.assertEqual(freq, "W")

    def test_top_rules_counts_and_shares(self):
        table = metrics.top_rules(self.cards, n=2)
        self.assertEqual(table.index[0], "R1")
        self.assertEqual(table.loc["R1", "total"], 4)
        self.assertAlmostEqual(table.loc["R1", "share"], 4 / 8)
        kpis = metrics.rules_kpis(self.cards, n=2)
        self.assertEqual(kpis["quiet_rules"], 1)  # R2: реальных инцидентов меньше 1 %

    def test_ecdf_is_monotone_and_reaches_one(self):
        values = np.sort(np.array([0.0, 5.0, 60.0, 3600.0]))
        grid = np.geomspace(1, 7200, 50)
        curve = metrics.ecdf(values, grid)
        self.assertTrue(np.all(np.diff(curve) >= 0))
        self.assertEqual(curve[-1], 1.0)
        self.assertEqual(curve[0], 0.25)  # мгновенные карточки видны в начале кривой

    def test_reaction_medians_ignore_negative_intervals(self):
        react = metrics.reaction_kpis(self.cards)
        self.assertEqual(react["known"], 6)
        self.assertEqual(react["median_critical"], (600 + 30) / 2)

    def test_sla_matrix_hides_small_cells(self):
        share, counts = metrics.sla_matrix(self.cards)
        self.assertTrue(share.isna().all().all())  # в каждой ячейке меньше 5 карточек
        self.assertEqual(int(counts.to_numpy().sum()), int(self.cards["sla_breach"].notna().sum()))


class ViewTests(unittest.TestCase):
    def test_every_view_renders_for_default_and_empty_filters(self):
        data = dataset()
        for f in (Filters(), Filters(severities=("Отсутствует", "Низкая"))):
            values = views.compute(data, f)
            for key in ("fig_flow", "fig_rules", "fig_react", "fig_sla"):
                self.assertIn(key, values)
            for key in views.TILES:
                self.assertTrue(values[f"{key}_v"])

    def test_russian_number_format(self):
        self.assertEqual(loader.ru_int(240295), "240 295")
        self.assertEqual(loader.ru_pct(0.627), "62,7 %")
        self.assertEqual(loader.ru_duration(0.2), "< 1 с")
        self.assertEqual(loader.ru_duration(312), "5,2 мин")

    def test_outcome_colors_are_shared_by_flow_and_rules(self):
        cards = dataset().cards
        dates = pd.date_range(cards["date"].min(), cards["date"].max())
        counts, freq = metrics.flow_by_time(cards, dates)
        flow = charts.flow(counts, freq)
        rules = charts.rules(metrics.top_rules(cards))
        colors = lambda fig: {t.name: t.marker.color for t in fig.data}
        self.assertEqual(colors(flow), colors(rules))


if __name__ == "__main__":
    unittest.main()
