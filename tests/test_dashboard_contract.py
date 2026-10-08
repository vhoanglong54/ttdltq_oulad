from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

import pandas as pd
from shapely.geometry import shape


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))

from dashboard_data import (  # noqa: E402
    compute_kpis,
    filter_attempts,
    load_dashboard_mart,
)


class DashboardCalculationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "code_module": ["AAA", "AAA", "BBB"],
                "code_presentation": ["2014J", "2014B", "2014J"],
                "id_student": [1, 1, 2],
                "region": ["R1", "R2", "R1"],
                "gender": ["F", "M", "F"],
                "At_Risk": [0, 1, 1],
                "not_complete": [0, 1, 0],
                "assessment_scored_count": [2, 1, 0],
                "assessment_score_sum_all_time": [160.0, 50.0, 0.0],
                "vle_total_clicks_all_time": [100, 20, 30],
            }
        )

    def test_weighted_kpis_use_attempt_and_learner_grains(self) -> None:
        result = compute_kpis(self.frame)
        self.assertEqual(result.attempts, 3)
        self.assertEqual(result.learners, 2)
        self.assertEqual(result.at_risk_count, 2)
        self.assertAlmostEqual(result.at_risk_rate, 2 / 3)
        self.assertEqual(result.not_complete_count, 1)
        self.assertAlmostEqual(result.not_complete_rate, 1 / 3)
        self.assertAlmostEqual(result.average_assessment_score, 210 / 3)
        self.assertEqual(result.vle_total_clicks, 150)

    def test_cascading_filter_function_keeps_original_unchanged(self) -> None:
        filtered = filter_attempts(
            self.frame,
            modules=["AAA"],
            presentations=["2014J"],
            regions=["R1"],
            genders=["F"],
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(int(filtered.iloc[0]["id_student"]), 1)
        self.assertEqual(len(self.frame), 3)


class DashboardMartTests(unittest.TestCase):
    def test_marts_support_meaningful_learner_context_filters(self) -> None:
        context = {"gender", "age_band", "highest_education", "imd_band", "region"}
        daily = load_dashboard_mart(
            "vle_daily_profile.csv.gz", context | {"At_Risk", "date", "sum_click"}
        )
        activity = load_dashboard_mart(
            "vle_activity_summary_day105.csv.gz",
            context | {"At_Risk", "activity_type", "sum_click"},
        )
        submissions = load_dashboard_mart(
            "assessment_submissions.csv.gz",
            context | {"At_Risk", "submission_delay", "score"},
        )
        self.assertFalse(daily.empty)
        self.assertFalse(activity.empty)
        self.assertFalse(submissions.empty)

    def test_vle_marts_preserve_all_clean_clicks(self) -> None:
        required = {
            "code_module",
            "code_presentation",
            "gender",
            "region",
            "sum_click",
        }
        daily = load_dashboard_mart(
            "vle_daily_profile.csv.gz", required | {"At_Risk", "date"}
        )
        activity = load_dashboard_mart(
            "vle_activity_summary.csv.gz", required | {"activity_type"}
        )
        cutoff_activity = load_dashboard_mart(
            "vle_activity_summary_day105.csv.gz",
            required | {"At_Risk", "activity_type"},
        )
        expected = int(
            pd.read_csv(
                ROOT / "data" / "processed" / "clean_dataset.csv",
                usecols=["vle_total_clicks_all_time"],
            )["vle_total_clicks_all_time"].sum()
        )
        self.assertEqual(int(daily["sum_click"].sum()), expected)
        self.assertEqual(int(activity["sum_click"].sum()), expected)
        self.assertEqual(
            int(cutoff_activity["sum_click"].sum()),
            int(daily.loc[daily["date"].le(105), "sum_click"].sum()),
        )

    def test_submission_delay_is_derived_from_due_date(self) -> None:
        submissions = load_dashboard_mart(
            "assessment_submissions.csv.gz",
            {"date_submitted", "due_date", "submission_delay", "score"},
        )
        expected = submissions["date_submitted"] - submissions["due_date"]
        pd.testing.assert_series_equal(
            submissions["submission_delay"].astype(float),
            expected.astype(float),
            check_names=False,
        )
        self.assertTrue(submissions["score"].between(0, 100).all())


class EvidenceContractTests(unittest.TestCase):
    def test_every_dashboard_visual_has_an_outcome_question(self) -> None:
        inventory = (ROOT / "dashboard" / "chart-inventory.md").read_text(
            encoding="utf-8"
        )
        rows = [
            line
            for line in inventory.splitlines()
            if re.match(r"^\|\s*\d+\s*\|", line)
        ]
        self.assertEqual(len(rows), 13)
        self.assertTrue(all("→" in row or "↔" in row for row in rows))

        app_source = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
        for removed_visual in (
            "px.treemap",
            "go.Indicator",
            "render_error_donut",
            "render_action_table",
            "outcome_drill_module",
        ):
            self.assertNotIn(removed_visual, app_source)

    def test_eda_outputs_cover_eight_insights_and_ten_hypotheses(self) -> None:
        insights = pd.read_csv(ROOT / "reports" / "eda" / "insight_evidence.csv")
        hypotheses = pd.read_csv(ROOT / "reports" / "eda" / "hypothesis_results.csv")
        self.assertEqual(set(insights["insight_id"]), {f"INS-{i:02d}" for i in range(1, 9)})
        self.assertEqual(set(hypotheses["hypothesis"]), {f"H{i:02d}" for i in range(1, 11)})
        self.assertTrue((insights["group_a_n"] > 0).all())
        self.assertTrue((insights["group_b_n"] > 0).all())
        self.assertTrue(insights["group_a_rate"].between(0, 1).all())
        self.assertTrue(insights["group_b_rate"].between(0, 1).all())

    def test_missing_assessment_score_is_explicit_and_h06_uses_q1_vs_q4(self) -> None:
        scores = pd.read_csv(
            ROOT / "reports" / "eda" / "assessment_score_quartiles.csv"
        )
        self.assertFalse(scores["assessment_score_quartile"].isna().any())
        self.assertIn(
            "No scored assessment by cutoff",
            set(scores["assessment_score_quartile"]),
        )
        indexed = scores.set_index("assessment_score_quartile")
        difference = (
            indexed.loc["Q1 — Lowest", "at_risk_rate"]
            - indexed.loc["Q4 — Highest", "at_risk_rate"]
        ) * 100
        hypotheses = pd.read_csv(
            ROOT / "reports" / "eda" / "hypothesis_results.csv"
        ).set_index("hypothesis")
        self.assertIn(f"{difference:.2f} pp", hypotheses.loc["H06", "evidence"])

    def test_geojson_matches_all_oulad_region_labels(self) -> None:
        geojson_path = ROOT / "dashboard" / "assets" / "oulad_regions.geojson"
        with geojson_path.open(encoding="utf-8") as handle:
            geojson = json.load(handle)
        geo_labels = {
            feature["properties"]["region"] for feature in geojson["features"]
        }
        data_labels = set(
            pd.read_csv(
                ROOT / "data" / "processed" / "clean_dataset.csv",
                usecols=["region"],
            )["region"].dropna()
        )
        self.assertEqual(len(geojson["features"]), 13)
        self.assertEqual(geo_labels, data_labels)
        self.assertEqual(
            geojson["metadata"]["geometry_license"],
            "Open Government Licence v3.0",
        )
        geometries = [shape(feature["geometry"]) for feature in geojson["features"]]
        self.assertTrue(all(geometry.is_valid for geometry in geometries))
        self.assertTrue(all(not geometry.is_empty for geometry in geometries))

    def test_geojson_mapping_audit_has_unique_ons_areas(self) -> None:
        audit = pd.read_csv(
            ROOT / "dashboard" / "assets" / "oulad_regions_mapping.csv"
        )
        self.assertFalse(audit["ons_code"].duplicated().any())
        self.assertEqual(audit["oulad_region"].nunique(), 13)
        self.assertFalse(audit["mapping_rationale"].isna().any())


if __name__ == "__main__":
    unittest.main()
