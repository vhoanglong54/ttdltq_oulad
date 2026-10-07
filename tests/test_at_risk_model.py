from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from at_risk_features import (  # noqa: E402
    MODEL_CATEGORICAL_FEATURES,
    MODEL_INTERACTION_FEATURES,
    MODEL_NUMERIC_FEATURES,
    PROHIBITED_MODEL_FEATURES,
    _assessment_features,
    build_base_cohort,
)
from at_risk_model import (  # noqa: E402
    CONFIDENCE_METRICS,
    VERIFICATION_THRESHOLDS,
    _group_bootstrap_confidence_intervals,
    choose_threshold,
    create_group_splits,
    make_error_type,
    make_risk_band,
)
from oulad_pipeline import VLE_EVENT_KEY, clean_frame  # noqa: E402


class DataPipelineTests(unittest.TestCase):
    def test_student_vle_consolidates_event_key_without_losing_clicks(self) -> None:
        raw = pd.DataFrame(
            {
                "code_module": ["AAA", "AAA", "AAA", "AAA"],
                "code_presentation": ["2014J"] * 4,
                "id_student": [1, 1, 1, 2],
                "id_site": [10, 10, 10, 11],
                "date": [5, 5, 5, 6],
                "sum_click": [2, 2, 3, 4],
            }
        )

        cleaned, metrics = clean_frame("studentVle.csv", raw)

        self.assertFalse(cleaned[VLE_EVENT_KEY].duplicated().any())
        self.assertEqual(int(raw["sum_click"].sum()), int(cleaned["sum_click"].sum()))
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(
            int(cleaned.loc[cleaned["id_student"].eq(1), "sum_click"].iloc[0]), 7
        )
        self.assertEqual(metrics["exact_duplicate_rows_observed"], 1)
        self.assertEqual(metrics["consolidated_event_key_rows"], 2)
        self.assertEqual(metrics["dropped_exact_duplicates"], 0)

    def test_weighted_assessment_features_exclude_post_cutoff_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pd.DataFrame(
                {
                    "code_module": ["AAA", "AAA"],
                    "code_presentation": ["2014J", "2014J"],
                    "id_assessment": [10, 11],
                    "assessment_type": ["TMA", "TMA"],
                    "date": [20, 80],
                    "weight": [40.0, 60.0],
                }
            ).to_csv(directory / "assessments.csv", index=False)
            pd.DataFrame(
                {
                    "id_assessment": [10, 11],
                    "id_student": [1, 1],
                    "date_submitted": [15, 110],
                    "is_banked": [0, 0],
                    "score": [80.0, 100.0],
                }
            ).to_csv(directory / "studentAssessment.csv", index=False)

            features = _assessment_features(directory, cutoff_day=105).iloc[0]

            self.assertEqual(int(features["assessment_submission_count_cutoff"]), 1)
            self.assertAlmostEqual(
                float(features["assessment_due_submitted_weight_cutoff"]), 40.0
            )
            self.assertAlmostEqual(
                float(features["assessment_due_weighted_points_cutoff"]), 32.0
            )
            self.assertAlmostEqual(
                float(features["assessment_weighted_score_cutoff"]), 80.0
            )


class CohortTests(unittest.TestCase):
    def test_cohort_excludes_future_registration_and_known_withdrawal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pd.DataFrame(
                {
                    "code_module": ["AAA"] * 4,
                    "code_presentation": ["2014J"] * 4,
                    "id_student": [1, 2, 3, 4],
                    "gender": ["F", "M", "F", "M"],
                    "region": ["R"] * 4,
                    "highest_education": ["A Level"] * 4,
                    "imd_band": ["10-20"] * 4,
                    "age_band": ["0-35"] * 4,
                    "num_of_prev_attempts": [0, 0, 1, 0],
                    "studied_credits": [60] * 4,
                    "disability": ["N"] * 4,
                    "final_result": ["Pass", "Fail", "Withdrawn", "Distinction"],
                }
            ).to_csv(directory / "studentInfo.csv", index=False)
            pd.DataFrame(
                {
                    "code_module": ["AAA"] * 4,
                    "code_presentation": ["2014J"] * 4,
                    "id_student": [1, 2, 3, 4],
                    "date_registration": [-5, 30, -10, np.nan],
                    "date_unregistration": [np.nan, np.nan, 20, np.nan],
                }
            ).to_csv(directory / "studentRegistration.csv", index=False)
            pd.DataFrame(
                {
                    "code_module": ["AAA"],
                    "code_presentation": ["2014J"],
                    "module_presentation_length": [250],
                }
            ).to_csv(directory / "courses.csv", index=False)
            pd.DataFrame(
                columns=[
                    "code_module",
                    "code_presentation",
                    "id_assessment",
                    "assessment_type",
                    "date",
                    "weight",
                ]
            ).to_csv(directory / "assessments.csv", index=False)
            pd.DataFrame(
                columns=["id_assessment", "id_student", "date_submitted", "is_banked", "score"]
            ).to_csv(directory / "studentAssessment.csv", index=False)
            pd.DataFrame(
                columns=[
                    "id_site",
                    "code_module",
                    "code_presentation",
                    "activity_type",
                    "week_from",
                    "week_to",
                ]
            ).to_csv(directory / "vle.csv", index=False)
            pd.DataFrame(
                columns=[
                    "code_module",
                    "code_presentation",
                    "id_student",
                    "id_site",
                    "date",
                    "sum_click",
                ]
            ).to_csv(directory / "studentVle.csv", index=False)

            cohort, counts = build_base_cohort(directory, cutoff_day=28)

            self.assertEqual(set(cohort["id_student"]), {1, 4})
            self.assertEqual(counts["excluded_registered_after_cutoff"], 1)
            self.assertEqual(counts["excluded_unregistered_by_cutoff"], 1)
            self.assertEqual(counts["excluded_withdrawn_outcome"], 1)
            self.assertEqual(dict(zip(cohort["id_student"], cohort["At_Risk"])), {1: 0, 4: 0})
            self.assertEqual(
                dict(zip(cohort["id_student"], cohort["Academic_Fail"])),
                {1: 0, 4: 0},
            )


class SplitAndThresholdTests(unittest.TestCase):
    def test_verification_policy_has_explicit_quality_gates(self) -> None:
        self.assertEqual(
            set(VERIFICATION_THRESHOLDS),
            {
                "minimum_accuracy",
                "minimum_accuracy_ci_lower",
                "minimum_recall",
                "minimum_balanced_accuracy",
                "minimum_f1",
                "minimum_roc_auc",
                "minimum_pr_auc_margin",
                "maximum_brier",
                "minimum_baseline_margin",
                "minimum_presentation_accuracy",
            },
        )
        self.assertGreaterEqual(VERIFICATION_THRESHOLDS["minimum_accuracy"], 0.80)
        self.assertGreaterEqual(
            VERIFICATION_THRESHOLDS["minimum_accuracy_ci_lower"], 0.80
        )
        self.assertGreaterEqual(VERIFICATION_THRESHOLDS["minimum_recall"], 0.70)
        self.assertGreaterEqual(
            VERIFICATION_THRESHOLDS["minimum_balanced_accuracy"], 0.78
        )
        self.assertGreaterEqual(VERIFICATION_THRESHOLDS["minimum_roc_auc"], 0.85)
        self.assertLessEqual(VERIFICATION_THRESHOLDS["maximum_brier"], 0.15)

    def test_student_groups_never_cross_splits(self) -> None:
        snapshot = pd.DataFrame(
            {
                "id_student": np.arange(1, 29),
                "At_Risk": [0, 1] * 14,
            }
        )
        split = create_group_splits(snapshot, random_state=42)
        self.assertFalse(split.isna().any())
        student_sets = {
            name: set(snapshot.loc[split.eq(name), "id_student"])
            for name in ("train", "validation", "test")
        }
        self.assertFalse(student_sets["train"] & student_sets["validation"])
        self.assertFalse(student_sets["train"] & student_sets["test"])
        self.assertFalse(student_sets["validation"] & student_sets["test"])

    def test_threshold_respects_recall_floor(self) -> None:
        actual = np.array([0, 0, 0, 1, 1, 1])
        probability = np.array([0.05, 0.20, 0.55, 0.45, 0.70, 0.90])
        threshold, table = choose_threshold(actual, probability, minimum_recall=0.66)
        selected = table.loc[table["selected"]].iloc[0]
        eligible = table.loc[table["meets_minimum_recall"]]
        self.assertAlmostEqual(threshold, float(selected["threshold"]))
        self.assertGreaterEqual(float(selected["recall"]), 0.66)
        self.assertAlmostEqual(float(selected["accuracy"]), float(eligible["accuracy"].max()))
        self.assertEqual(len(table), 901)

    def test_risk_bands_and_error_types_are_explicit(self) -> None:
        probabilities = np.array([0.10, 0.30, 0.70])
        self.assertEqual(list(make_risk_band(probabilities, threshold=0.60)), ["Low", "Medium", "High"])
        actual = np.array([1, 0, 0, 1])
        predicted = np.array([1, 0, 1, 0])
        self.assertEqual(list(make_error_type(actual, predicted)), ["TP", "TN", "FP", "FN"])

    def test_feature_contract_excludes_target_and_identifiers(self) -> None:
        features = set(
            MODEL_CATEGORICAL_FEATURES
            + MODEL_NUMERIC_FEATURES
            + MODEL_INTERACTION_FEATURES
        )
        self.assertFalse(features & PROHIBITED_MODEL_FEATURES)
        self.assertTrue(
            {
                "assessment_weight_completion_rate_cutoff",
                "assessment_due_performance_rate_cutoff",
                "vle_recent_28_click_share",
                "module_AAA_x_assessment_weighted_score_cutoff",
            }.issubset(features)
        )

    def test_group_bootstrap_confidence_intervals_are_well_formed(self) -> None:
        predictions = pd.DataFrame(
            {
                "id_student": np.arange(1, 41),
                "dataset_split": ["test"] * 40,
                "actual_at_risk": [0, 1] * 20,
                "risk_probability": np.tile([0.15, 0.85], 20),
            }
        )
        intervals = _group_bootstrap_confidence_intervals(
            predictions,
            threshold=0.5,
            model_version="test-v1",
            iterations=50,
            random_state=7,
        )

        self.assertEqual(set(intervals["metric"]), set(CONFIDENCE_METRICS))
        self.assertTrue(intervals["bootstrap_unit"].eq("id_student").all())
        self.assertTrue(
            (intervals["lower_95"] <= intervals["point_estimate"]).all()
        )
        self.assertTrue(
            (intervals["point_estimate"] <= intervals["upper_95"]).all()
        )


if __name__ == "__main__":
    unittest.main()
