"""Проверки расчётов задания 2 без parquet и без запуска Taipy.

Запуск из корня репозитория:
    poetry run python -m unittest discover -s task_2
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import charts
import loader
import metrics
import views
from loader import Data, Filters


def raw_predictions() -> pd.DataFrame:
    scores = [0.95, 0.70, 0.40, 0.25, 0.10, 0.05, 0.90, 0.20, 0.01, 0.30]
    return pd.DataFrame({
        "id": range(1, 11),
        "assessed": pd.to_datetime(["2025-09-01 10:00", "2025-09-02 10:00", "2025-09-08 10:00",
                                    "2025-09-09 10:00", "2025-09-15 10:00", "2025-10-01 10:00",
                                    "2025-10-02 10:00", "2025-10-03 10:00", "2025-10-04 10:00",
                                    "2025-10-05 22:30"], utc=True),
        "label": [1, 1, 0, 1, 1, 0, 1, 0, 0, np.nan],
        "predicted": [int(s >= 0.18) for s in scores],
        "score": scores,
        "risk": [4, 3, 2, 1, 1, 1, 4, 1, 1, 1],
        "severity": ["Высокая", "Средняя", "Средняя", "Высокая", "Высокая", None, "Средняя", "Высокая", "Средняя",
                     "Средняя"],
        "category": ["Управление УЗ", "Инцидент", "Инцидент", "Управление УЗ", "Управление УЗ", "Инцидент",
                     "Инцидент", "Инцидент", "Инцидент", "Инцидент"],
        "vendor": ["X", None, "", "X", None, "X", "[]", "X", None, "X"],
    })


def history_raw() -> pd.DataFrame:
    return pd.DataFrame({
        "Создан": ["2025-01-15T10:00:00+03:00"] * 4 + ["2025-10-15T10:00:00+03:00"] * 4,
        "Вердикт": ["Решён", "False Positive", "False Positive", None, "Решён", "Решён", "False Positive", "SPAM"],
        "Критичность": ["Высокая"] * 8,
        "Вендор из сием": ["V", "V", None, "V", None, None, "V", None],
    })


def data() -> Data:
    predictions, excluded = loader.prepare_predictions(raw_predictions())
    return Data(predictions=predictions, excluded=excluded,
                history=loader.prepare_history(history_raw()), path=Path("synthetic.parquet"))


class LoaderTests(unittest.TestCase):
    def test_only_resolved_and_fp_verdicts_enter_metrics(self):
        d = data()
        self.assertEqual(len(d.predictions), 9)
        self.assertEqual(d.excluded, 1)

    def test_outcomes_follow_threshold(self):
        outcomes = dict(zip(data().predictions["id"], data().predictions["outcome"]))
        self.assertEqual(outcomes[1], "TP")
        self.assertEqual(outcomes[5], "FN")  # реальный инцидент с оценкой 0,10
        self.assertEqual(outcomes[8], "FP")  # 0,20 ≥ 0,18, но False Positive
        self.assertEqual(outcomes[9], "TN")

    def test_threshold_mismatch_is_rejected(self):
        raw = raw_predictions()
        raw.loc[0, "predicted"] = 0
        with self.assertRaises(ValueError):
            loader.prepare_predictions(raw)

    def test_empty_vendor_values_are_not_filled(self):
        vendor = dict(zip(data().predictions["id"], data().predictions["vendor_filled"]))
        self.assertTrue(vendor[1])
        self.assertFalse(vendor[3])  # пустая строка
        self.assertFalse(vendor[7])  # пустой JSON-массив

    def test_history_counts_verdicts_and_vendor_by_month(self):
        h = data().history.set_index("month")
        self.assertEqual(int(h.loc["2025-01", "real"]), 1)
        self.assertEqual(int(h.loc["2025-01", "fp"]), 2)
        self.assertEqual(int(h.loc["2025-10", "vendor"]), 1)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.frame = data().predictions

    def test_confusion_and_miss_share(self):
        c = metrics.confusion(self.frame)
        self.assertEqual((c["TP"], c["FN"], c["FP"], c["TN"]), (4, 1, 2, 2))
        self.assertAlmostEqual(c["miss"], 1 / 5)
        self.assertAlmostEqual(c["precision"], 4 / 6)
        self.assertLessEqual(c["miss_low"], c["miss"])
        self.assertGreaterEqual(c["miss_high"], c["miss"])

    def test_wilson_interval_is_inside_zero_one(self):
        self.assertEqual(metrics.wilson(0, 10)[0], 0.0)
        self.assertEqual(metrics.wilson(10, 10)[1], 1.0)
        self.assertTrue(np.isnan(metrics.wilson(0, 0)[0]))

    def test_weekly_miss_marks_small_weeks(self):
        table = metrics.weekly_miss(self.frame)
        self.assertEqual(int(table["incidents"].sum()), 5)
        self.assertFalse(table["reliable"].any())  # меньше 10 инцидентов в каждой неделе

    def test_gains_curve_reaches_all_incidents(self):
        curve = metrics.gains(self.frame)
        self.assertEqual(curve["found"][-1], 1.0)
        self.assertTrue(np.all(np.diff(curve["found"]) >= 0))
        qx, qy = curve["threshold"]
        self.assertAlmostEqual(qx, 6 / 9)
        self.assertAlmostEqual(qy, 4 / 5)

    def test_category_miss_respects_minimum(self):
        self.assertTrue(metrics.category_miss(self.frame).empty)
        table = metrics.category_miss(self.frame, min_incidents=1)
        self.assertAlmostEqual(table.loc["Управление учётными записями", "share"], 1 / 3)

    def test_risk_rows_split_low_level_by_threshold(self):
        rows = metrics.risk_rows(self.frame).set_index("row")
        self.assertEqual(int(rows.iloc[0]["cards"]), 3)  # оценки ниже 0,18
        self.assertEqual(int(rows.iloc[1]["cards"]), 2)  # 0,18–0,37: «низкий», но в приоритете
        self.assertEqual(int(rows["cards"].sum()), len(self.frame))
        self.assertAlmostEqual(rows["of_all_real"].sum(), 1.0)

    def test_history_shares(self):
        monthly = metrics.history_monthly(data().history).set_index("month")
        self.assertAlmostEqual(monthly.loc["2025-01", "real_share"], 1 / 3)
        self.assertAlmostEqual(monthly.loc["2025-10", "real_share"], 2 / 3)
        self.assertAlmostEqual(monthly.loc["2025-01", "vendor_share"], 3 / 4)


class ViewTests(unittest.TestCase):
    def test_every_view_renders_for_default_and_narrow_filters(self):
        d = data()
        for f in (Filters(), Filters("2025-10"), Filters(severities=("Отсутствует",))):
            values = views.compute(d, f)
            for key in ("fig_weekly", "fig_gains", "fig_categories", "fig_risk", "fig_history"):
                self.assertIn(key, values)
            for key in views.TILES:
                self.assertTrue(values[f"{key}_v"])

    def test_missing_history_shows_hint_instead_of_chart(self):
        d = data()
        d = Data(predictions=d.predictions, excluded=d.excluded, history=None, path=d.path)
        values = views.compute(d, Filters())
        self.assertIn("iavd_ml_dataset", values["fig_history"].layout.annotations[0].text)

    def test_missed_incidents_share_one_colour(self):
        frame = data().predictions
        weekly = charts.weekly(metrics.weekly_miss(frame), metrics.confusion(frame)["miss"])
        cats = charts.categories(metrics.category_miss(frame, min_incidents=1), 0.2)
        self.assertEqual(cats.data[0].marker.color, charts.MISSED)
        self.assertIn(charts.AXIS, {t.marker.color for t in weekly.data})  # недели с малым числом инцидентов


if __name__ == "__main__":
    unittest.main()
