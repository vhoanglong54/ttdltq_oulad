"""Data contracts and reusable calculations for the Streamlit dashboard.

The UI must only consume attempt-level tables produced by the verified Python
pipelines.  This module deliberately contains no Streamlit state so its
calculations can be unit-tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = PROJECT_ROOT / "data" / "processed" / "clean_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "model_academic_fail" / "c105_final"
DASHBOARD_DIR = PROJECT_ROOT / "data" / "processed" / "dashboard"
SNAPSHOT_PATH = MODEL_DIR / "feature_snapshot.csv"
EDA_DIR = PROJECT_ROOT / "reports" / "eda"
REGION_GEOJSON_PATH = PROJECT_ROOT / "dashboard" / "assets" / "oulad_regions.geojson"

ATTEMPT_KEY = ["code_module", "code_presentation", "id_student"]
ANALYSIS_REQUIRED_COLUMNS = {
    *ATTEMPT_KEY,
    "region",
    "final_result",
    "At_Risk",
    "assessment_scored_count",
    "assessment_score_sum_all_time",
    "assessment_score_mean_all_time",
    "vle_total_clicks_all_time",
    "vle_active_days_all_time",
}
PREDICTION_REQUIRED_COLUMNS = {
    *ATTEMPT_KEY,
    "region",
    "dataset_split",
    "actual_at_risk",
    "actual_fail",
    "actual_status",
    "risk_probability",
    "failure_probability",
    "predicted_at_risk",
    "predicted_fail",
    "predicted_status",
    "risk_band",
    "error_type",
    "cutoff_day",
    "prediction_threshold",
    "model_version",
}
SNAPSHOT_REQUIRED_COLUMNS = {
    *ATTEMPT_KEY,
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
    "final_result",
    "At_Risk",
    "actual_status",
    "dataset_split",
    "num_of_prev_attempts",
    "studied_credits",
    "assessment_weighted_score_cutoff",
    "assessment_completion_rate_cutoff",
    "vle_total_clicks_cutoff",
    "vle_active_days_cutoff",
    "vle_clicks_last_7_days",
    "vle_clicks_previous_7_days",
    "vle_clicks_previous_28_days",
    "vle_clicks_last_28_days",
}


@dataclass(frozen=True)
class DashboardKpis:
    """Unambiguous KPI values for the current filter context."""

    attempts: int
    learners: int
    at_risk_count: int
    at_risk_rate: float
    average_assessment_score: float
    vle_total_clicks: int


def _read_required_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file bắt buộc: {path}")

    frame = pd.read_csv(path)
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} thiếu cột bắt buộc: {', '.join(missing)}")
    return frame


def _assert_attempt_grain(frame: pd.DataFrame, source_name: str) -> None:
    duplicate_count = int(frame.duplicated(ATTEMPT_KEY).sum())
    if duplicate_count:
        raise ValueError(
            f"{source_name} có {duplicate_count:,} khóa attempt trùng; "
            "dashboard dừng để tránh nhân sai KPI."
        )


def load_analysis_data() -> pd.DataFrame:
    """Load the canonical descriptive table and enforce its attempt grain."""

    frame = _read_required_csv(ANALYSIS_PATH, ANALYSIS_REQUIRED_COLUMNS)
    frame["Academic_Fail"] = frame["final_result"].eq("Fail").astype("int8")
    frame["Withdrawn_Flag"] = frame["final_result"].eq("Withdrawn").astype("int8")
    frame["Academic_Success"] = frame["final_result"].isin(
        ["Pass", "Distinction"]
    ).astype("int8")
    # Keep the existing calculation API stable while changing the outcome to
    # the approved academic target: Fail only, not Fail + Withdrawn.
    frame["At_Risk"] = frame["Academic_Fail"]
    _assert_attempt_grain(frame, ANALYSIS_PATH.name)
    return frame


def load_model_table(filename: str, required_columns: set[str] | None = None) -> pd.DataFrame:
    """Load one generated model artifact without silently fabricating defaults."""

    path = MODEL_DIR / filename
    return _read_required_csv(path, required_columns or set())


def load_predictions() -> pd.DataFrame:
    """Load prediction rows and enforce one row per eligible attempt."""

    frame = load_model_table("model_predictions.csv", PREDICTION_REQUIRED_COLUMNS)
    _assert_attempt_grain(frame, "model_predictions.csv")
    return frame


def load_feature_snapshot() -> pd.DataFrame:
    """Load cutoff-safe analysis features at one row per eligible attempt."""

    frame = _read_required_csv(SNAPSHOT_PATH, SNAPSHOT_REQUIRED_COLUMNS)
    _assert_attempt_grain(frame, SNAPSHOT_PATH.name)
    return frame


def load_eda_table(filename: str, required_columns: set[str] | None = None) -> pd.DataFrame:
    """Load a generated EDA evidence table with an optional schema contract."""

    return _read_required_csv(EDA_DIR / filename, required_columns or set())


def load_dashboard_mart(
    filename: str, required_columns: set[str] | None = None
) -> pd.DataFrame:
    """Load a compact chart-specific mart built by ``dashboard_features.py``."""

    return _read_required_csv(DASHBOARD_DIR / filename, required_columns or set())


def filter_attempts(
    frame: pd.DataFrame,
    modules: list[str] | None = None,
    presentations: list[str] | None = None,
    regions: list[str] | None = None,
    genders: list[str] | None = None,
) -> pd.DataFrame:
    """Apply the shared dashboard filters without changing the input frame."""

    mask = pd.Series(True, index=frame.index)
    if modules:
        mask &= frame["code_module"].isin(modules)
    if presentations:
        mask &= frame["code_presentation"].isin(presentations)
    if regions:
        mask &= frame["region"].isin(regions)
    if genders:
        mask &= frame["gender"].isin(genders)
    return frame.loc[mask].copy()


def compute_kpis(frame: pd.DataFrame) -> DashboardKpis:
    """Compute KPIs using the definitions in the processed data contract."""

    attempts = len(frame)
    learners = int(frame["id_student"].nunique()) if attempts else 0
    at_risk_count = int(frame["At_Risk"].sum()) if attempts else 0
    at_risk_rate = at_risk_count / attempts if attempts else float("nan")

    scored_count = float(frame["assessment_scored_count"].fillna(0).sum())
    score_sum = float(frame["assessment_score_sum_all_time"].fillna(0).sum())
    average_score = score_sum / scored_count if scored_count else float("nan")
    total_clicks = int(frame["vle_total_clicks_all_time"].fillna(0).sum())

    return DashboardKpis(
        attempts=attempts,
        learners=learners,
        at_risk_count=at_risk_count,
        at_risk_rate=at_risk_rate,
        average_assessment_score=average_score,
        vle_total_clicks=total_clicks,
    )


def cascading_options(
    frame: pd.DataFrame,
    modules: list[str] | None = None,
    presentations: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return Module -> Presentation -> Region choices for cascading filters."""

    module_options = sorted(frame["code_module"].dropna().astype(str).unique())
    after_module = filter_attempts(frame, modules=modules)
    presentation_options = sorted(
        after_module["code_presentation"].dropna().astype(str).unique()
    )
    after_presentation = filter_attempts(
        after_module, presentations=presentations
    )
    region_options = sorted(after_presentation["region"].dropna().astype(str).unique())
    return module_options, presentation_options, region_options
