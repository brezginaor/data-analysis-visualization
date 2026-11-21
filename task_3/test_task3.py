"""Проверки расчётов задания 3 на синтетических данных, без запуска Taipy.

Запуск из корня репозитория:
    .venv/bin/python -m unittest discover -s task_3
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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

    def test_raw_score_additivity_exact(self):
        frame = shap_frame()
        frame["raw_score"] = BASE + frame[list(FEATURES)].sum(axis=1)
        self.assertAlmostEqual(
            loader.raw_score_additivity_check(frame, FEATURES, BASE), 0.0, places=9)

    def test_raw_score_additivity_mismatch(self):
        frame = shap_frame()
        frame["raw_score"] = BASE + frame[list(FEATURES)].sum(axis=1)
        frame.loc[frame.index[0], "raw_score"] += 0.01
        self.assertGreater(loader.raw_score_additivity_check(frame, FEATURES, BASE), 1e-6)

    def test_raw_score_missing(self):
        self.assertIsNone(loader.raw_score_additivity_check(shap_frame(), FEATURES, BASE))

    def test_apply_filters_by_team_and_outcome(self):
        frame = shap_frame()
        by_team = loader.apply_filters(frame, Filters(team=2))
        self.assertEqual(len(by_team), 4)
        by_outcome = loader.apply_filters(frame, Filters(outcomes=("TP",)))
        self.assertEqual(set(by_outcome["outcome"]), {"TP"})

    def test_missing_repo_csv_has_useful_error(self):
        with TemporaryDirectory() as tmp:
            with patch.object(loader, "CARDS_FILE", Path(tmp) / "cards_practice.csv"):
                with self.assertRaisesRegex(FileNotFoundError, "data/task3"):
                    loader.resolve_data_files()

    def test_raw_score_from_csv_is_not_a_shap_feature(self):
        frame = shap_frame().copy()
        frame["team"] = loader.PROJECT_TEAM
        frame["raw_score"] = BASE + frame[list(FEATURES)].sum(axis=1)
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            card_path = folder / "cards_practice.csv"
            shap_path = folder / "shap_cards.csv"
            frame[["EntityId", "team", "created", "y_true", "proba", "outcome", "flag"]].to_csv(
                card_path, sep=";", index=False)
            frame[["EntityId", "y_true", "proba", "outcome", "base", "raw_score", *FEATURES]].to_csv(
                shap_path, sep=";", index=False)
            data = loader.load(card_path, shap_path)
            self.assertEqual(data.features, FEATURES)
            self.assertIsNotNone(data.raw_check)
            self.assertLess(data.raw_check, loader.ADDITIVITY_TOLERANCE)

            frame.loc[frame.index[0], "raw_score"] += 0.01
            frame[["EntityId", "y_true", "proba", "outcome", "base", "raw_score", *FEATURES]].to_csv(
                shap_path, sep=";", index=False)
            with self.assertRaisesRegex(ValueError, "Аддитивность SHAP нарушена"):
                loader.load(card_path, shap_path)

    def test_repo_data_path_and_only_team_four(self):
        """Стандартные CSV из data/task3: отбирается только команда 4."""
        frame_2 = shap_frame()
        frame_4 = frame_2.copy()
        frame_4["EntityId"] += 100
        frame_4["team"] = loader.PROJECT_TEAM
        combined = pd.concat([frame_2, frame_4], ignore_index=True)
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "data" / "task3"
            folder.mkdir(parents=True)
            combined[["EntityId", "team", "created", "y_true", "proba", "outcome", "flag"]].to_csv(
                folder / "cards_practice.csv", sep=";", index=False)
            combined[["EntityId", "y_true", "proba", "outcome", "base", *FEATURES]].to_csv(
                folder / "shap_cards.csv", sep=";", index=False)
            with patch.object(loader, "CARDS_FILE", folder / "cards_practice.csv"), \
                 patch.object(loader, "SHAP_FILE", folder / "shap_cards.csv"):
                data = loader.load()
            self.assertEqual(data.path.parent, folder)
            self.assertEqual(len(data.cards), 4)
            self.assertEqual(len(data.shap), 4)
            self.assertEqual(set(data.cards["team"]), {loader.PROJECT_TEAM})
            self.assertEqual(set(data.shap["team"]), {loader.PROJECT_TEAM})
            labels = views.card_labels(data, loader.Filters(outcomes=("FP",)))
            self.assertEqual(len(labels), 1)
            self.assertTrue(labels[0].startswith("102 "))
            self.assertIsNone(data.raw_check)
            self.assertEqual(len(data.features), len(FEATURES))


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
        self.assertEqual(doubt["fp_near_total"], 1)  # единственный FP с оценкой ниже 10%
        self.assertEqual(doubt["fp_near"], 1)

    def test_fp_low_score_denominator_excludes_high_fp(self):
        flow = shap_frame()
        # Ещё один FP ниже 10%, но без пересечения порога при условном пересчёте.
        low = flow.iloc[1].copy()
        low["EntityId"] = 5
        low["Алерт"] = 0.0
        low["f_host"] = 1.0
        low["f_rare"] = np.log(0.05 / 0.95) - BASE - 1.0
        low["proba"] = 0.05
        # FP с оценкой 50% не должен попадать в знаменатель «ниже 10%».
        high = flow.iloc[1].copy()
        high["EntityId"] = 6
        high["Алерт"] = 0.0
        high["f_host"] = 2.0
        high["f_rare"] = 0.0
        high["proba"] = 0.5
        sample = pd.concat([flow, pd.DataFrame([low, high])], ignore_index=True)
        doubt = metrics.enrichment_doubt(sample, FEATURES)
        self.assertEqual(doubt["raised"], 5)
        self.assertEqual(doubt["fp_near_total"], 2)
        self.assertEqual(doubt["fp_near"], 1)
        self.assertEqual(doubt["on_enrichment"], 2)

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
            charts.doubt_comparison(
                float(flow.iloc[1]["proba"]),
                metrics.conditional_without_enrichment(flow.iloc[1], FEATURES), 2),
        )
        for fig in figures:
            self.assertIsInstance(fig, go.Figure)
            self.assertGreaterEqual(len(fig.data), 1)
        self.assertEqual(len(charts.empty().data), 0)

    def test_doubt_comparison_uses_hypothetical_shap_only(self):
        row = metrics.doubtful_case(shap_frame(), FEATURES)
        self.assertIsNotNone(row)
        conditional = metrics.conditional_without_enrichment(row, FEATURES)
        self.assertLess(conditional, THRESHOLD)
        fig = charts.doubt_comparison(float(row["proba"]), conditional, int(row["EntityId"]))
        self.assertEqual(list(fig.data[0].x), [float(row["proba"]), conditional])
        self.assertEqual(list(fig.data[0].y), ["Оценка модели", "Пересчёт SHAP без 3 вкладов"])
        self.assertIn(str(int(row["EntityId"])), fig.data[0].hovertemplate)
        self.assertFalse(any(str(int(row["EntityId"])) in str(a.text) for a in fig.layout.annotations))

    def test_waterfall_layout_has_readable_labels(self):
        row = shap_frame().iloc[0]
        wf = metrics.waterfall_values(row, FEATURES, BASE)
        fig = charts.waterfall(wf, float(row["proba"]))
        self.assertEqual(fig.data[0].orientation, "h")
        self.assertIn("Базовая линия", list(fig.data[0].y))
        self.assertEqual(len(fig.data[0].y), len(wf))
        self.assertEqual(list(fig.data[0].customdata), list(wf["feature"]))

    def test_waterfall_base_and_total_are_markers_not_bars(self):
        row = shap_frame().iloc[0]
        wf = metrics.waterfall_values(row, FEATURES, BASE)
        fig = charts.waterfall(wf, float(row["proba"]))
        self.assertEqual(fig.data[0].type, "waterfall")
        self.assertEqual(fig.data[1].type, "scatter")
        self.assertEqual(fig.data[1].mode, "markers+text")
        self.assertEqual(list(fig.data[1].x),
                         [float(wf.iloc[0]["value"]), float(wf.iloc[-1]["value"])])
        self.assertEqual(fig.data[0].totals.marker.color, "rgba(0,0,0,0)")
        self.assertEqual(fig.data[0].text[0], "")
        self.assertEqual(fig.data[0].text[-1], "")
        # Не включаем ноль в диапазон только ради пустых полос к базе и итогу.
        self.assertLess(fig.layout.xaxis.range[1], 0)

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

    def test_doubt_caption_not_bound_to_waterfall_selection(self):
        data = self.data()
        first = views.compute(data, Filters(team=2), 1)
        second = views.compute(data, Filters(team=2), 3)
        self.assertEqual(first["doubt_caption"], second["doubt_caption"])
        self.assertIn("карточка № 2", first["doubt_caption"])
        self.assertEqual(views.TILES["t_doubt"], "Условное снижение ниже порога")

    def test_summary_without_card(self):
        data = self.data()
        s = views.summary(data, Filters(), 1)
        self.assertIsNotNone(s["card"])
        self.assertEqual(s["card"]["entity"], 1)
        self.assertFalse(s["top"].empty)


class ReportChecks(unittest.TestCase):
    def test_report_does_not_claim_tp_proves_shap_or_base_means_training(self):
        import report

        frame = shap_frame()
        frame["team"] = loader.PROJECT_TEAM
        data = loader.Data(
            cards=frame[["EntityId", "team", "created", "y_true", "proba", "outcome", "flag"]],
            shap=frame, features=FEATURES, base=BASE, check=0.0, path=Path("unused"),
        )
        with patch.object(report.loader, "load", return_value=data):
            content = report.build(None, None)
        self.assertIn("Исход TP подтверждает правильность классификации", content)
        self.assertIn("не смысл SHAP-объяснения", content)
        self.assertIn("Базовое значение SHAP", content)
        self.assertNotIn("средний логарифм шансов по обучению", content)
        self.assertNotIn("деградировало с сентября", content)
        self.assertIn("группа из 4 карточек", content)


if __name__ == "__main__":
    unittest.main()
