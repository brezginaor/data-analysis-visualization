"""Проверки расчётов задания 3 без данных course_pack и без запуска Taipy.

Запуск из корня репозитория:
    .venv/bin/python -m unittest discover -s task_3
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import charts
import loader
import metrics
import views
from loader import Filters, THRESHOLD


FEATURES = ("Алерт", "f_host", "f_rare")  # «Алерт» — реальный признак обогащения из loader.ENRICHMENT
BASE = -2.0


def shap_frame() -> pd.DataFrame:
    """4 карточки: TP у порога на «Алерте», FP у порога на «Алерте», TP сильная, TN.

    proba каждой карточки равна sigmoid(base + ΣSHAP), как в данных курса.
    """
    logit_threshold = float(np.log(THRESHOLD / (1 - THRESHOLD)))
    rows = [
        # EntityId, Алерт, f_host, f_rare, outcome
        (1, 1.0, 1.0, logit_threshold - BASE - 2.0 + 0.01, "TP"),  # сумма чуть выше logit(τ); без «Алерта» — ниже порога
        (2, 1.0, 0.5, logit_threshold - BASE - 1.5 + 0.026, "FP"),
        (3, 0.1, 5.0, 0.4, "TP"),                            # сильная TP, обогащение не решает
        (4, -0.2, -4.0, -0.1, "TN"),
    ]
    frame = pd.DataFrame(rows, columns=["EntityId", "Алерт", "f_host", "f_rare", "outcome"])
    total = BASE + frame[["Алерт", "f_host", "f_rare"]].sum(axis=1)
    frame["proba"] = 1 / (1 + np.exp(-total))
    frame["y_true"] = frame["outcome"].isin(["TP", "FN"]).astype(int)
    frame["base"] = BASE
    frame["team"] = 2
    frame["created"] = "2025-09-01"
    frame["flag"] = np.where(frame["proba"] >= THRESHOLD, "инцидент", "FP")
    return frame


class DataLoaderChecks(unittest.TestCase):
    def test_ru_proba_tiny_value(self):
        self.assertEqual(loader.ru_proba(2e-06), "2,0·10⁻⁶")
        self.assertEqual(loader.ru_proba(0.0209), "0,0209")
        self.assertEqual(loader.ru_proba(0), "0")

    def test_reconstruction_check(self):
        frame = shap_frame()
        check = loader.reconstruction_check(frame, FEATURES, BASE)
        self.assertAlmostEqual(check, 0.0, places=9)  # proba согласована с base + ΣSHAP

    def test_apply_filters_by_team_and_outcome(self):
        frame = shap_frame()
        by_team = loader.apply_filters(frame, Filters(team=2))
        self.assertEqual(len(by_team), 4)
        by_outcome = loader.apply_filters(frame, Filters(outcomes=("TP",)))
        self.assertEqual(set(by_outcome["outcome"]), {"TP"})


class MetricsChecks(unittest.TestCase):
    def test_card_contributions_sorted_by_abs(self):
        row = shap_frame().iloc[0]
        table = metrics.card_contributions(row, FEATURES)
        self.assertEqual(table.iloc[0]["feature"], "f_rare")  # |−3,92| — самый большой вклад
        self.assertTrue(table["shap"].abs().is_monotonic_decreasing)

    def test_waterfall_sums_to_logit(self):
        row = shap_frame().iloc[0]
        table = metrics.waterfall_values(row, FEATURES, BASE, k=2)
        total = float(table.loc[table["measure"] == "total", "value"].iloc[0])
        self.assertAlmostEqual(total, BASE + row[list(FEATURES)].sum(), places=9)
        self.assertEqual(table.iloc[-1]["feature"], "Оценка модели (лог. шансов)")

    def test_mean_abs_top_and_share(self):
        table = metrics.mean_abs(shap_frame(), FEATURES, k=2)
        self.assertEqual(len(table), 2)
        self.assertAlmostEqual(table["share"].sum(), 1.0, places=9)
        # сильнее всего модель держится на f_host: средний |SHAP| максимален
        self.assertEqual(table.iloc[0]["feature"], "f_host")

    def test_enrichment_doubt_counterfactual(self):
        doubt = metrics.enrichment_doubt(shap_frame(), FEATURES)
        # подняты карточки 1, 2, 3; без Алерт (3.0 / 3.0 / 0.1 лог. шанса) 1 и 2 падают ниже порога
        self.assertEqual(doubt["raised"], 3)
        self.assertEqual(doubt["on_enrichment"], 2)
        self.assertAlmostEqual(doubt["share"], 2 / 3, places=9)
        self.assertEqual(doubt["fp_near_total"], 1)  # из поднятых FP только карточка 2
        self.assertEqual(doubt["fp_near"], 1)

    def test_confirmed_and_doubtful_cases(self):
        flow = shap_frame()
        confirmed = metrics.confirmed_case(flow, FEATURES)
        self.assertEqual(confirmed["EntityId"], 3)  # TP без опоры на обогащение
        doubtful = metrics.doubtful_case(flow, FEATURES)
        self.assertEqual(doubtful["EntityId"], 2)   # FP, который без Алерт ниже порога

    def test_empty_flow_is_safe(self):
        empty = shap_frame().iloc[0:0]
        self.assertTrue(metrics.mean_abs(empty, FEATURES).empty)
        doubt = metrics.enrichment_doubt(empty, FEATURES)
        self.assertEqual(doubt["raised"], 0)
        self.assertIsNone(metrics.confirmed_case(empty, FEATURES))
        self.assertIsNone(metrics.doubtful_case(empty, FEATURES))


class ChartsChecks(unittest.TestCase):
    def test_figures_build(self):
        import plotly.graph_objects as go

        flow = shap_frame()
        row = flow.iloc[0]
        wf = metrics.waterfall_values(row, FEATURES, BASE)
        figures = (
            charts.waterfall(wf, float(row["proba"])),
            charts.mean_abs(metrics.mean_abs(flow, FEATURES)),
            charts.signed_outcomes(metrics.signed_by_outcome(flow, FEATURES)),
            charts.alert_scatter(flow, "Алерт"),
        )
        for fig in figures:
            self.assertIsInstance(fig, go.Figure)
            self.assertGreaterEqual(len(fig.data), 1)
        self.assertEqual(len(charts.empty().data), 0)

    def test_waterfall_measure_labels(self):
        row = shap_frame().iloc[0]
        table = metrics.waterfall_values(row, FEATURES, BASE, k=1)
        self.assertEqual(list(table["measure"]), ["absolute", "relative", "relative", "total"])


class ViewsChecks(unittest.TestCase):
    def data(self) -> loader.Data:
        frame = shap_frame()
        return loader.Data(cards=frame[["EntityId", "team", "created", "y_true", "proba", "outcome", "flag"]],
                           shap=frame, features=FEATURES, base=BASE, check=1e-7,
                           path=Path("unused"))

    def test_compute_tiles(self):
        values = views.compute(self.data(), Filters(team=2), 2)
        for key in views.TILES:
            self.assertIn(f"{key}_v", values)
            self.assertIn(f"{key}_n", values)
        self.assertIn("fig_card", values)
        self.assertTrue(str(values["t_card_v"]))

    def test_summary_without_card(self):
        data = self.data()
        s = views.summary(data, Filters(), 1)
        self.assertIsNotNone(s["card"])
        self.assertEqual(s["card"]["entity"], 1)
        self.assertFalse(s["top"].empty)


if __name__ == "__main__":
    unittest.main()
