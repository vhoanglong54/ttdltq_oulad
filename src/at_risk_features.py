"""Build leakage-safe early-warning features from cleaned OULAD tables.

The functions in this module deliberately read the cleaned event tables from
``data/interim`` rather than the descriptive ``clean_dataset.csv``.  The
descriptive table contains all-time aggregates and is therefore not a valid
input for an early-warning model.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


ATTEMPT_KEY = ["code_module", "code_presentation", "id_student"]
REQUIRED_FILES = (
    "courses.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "assessments.csv",
    "studentAssessment.csv",
    "vle.csv",
    "studentVle.csv",
)

# Protected demographic attributes are retained in prediction exports for
# subgroup QA, but are not used by the primary model.
AUDIT_COLUMNS = ["gender", "region", "imd_band", "age_band", "disability"]
VLE_ACTIVITY_TYPES = (
    "dataplus",
    "dualpane",
    "externalquiz",
    "folder",
    "forumng",
    "glossary",
    "homepage",
    "htmlactivity",
    "oucollaborate",
    "oucontent",
    "ouelluminate",
    "ouwiki",
    "page",
    "questionnaire",
    "quiz",
    "repeatactivity",
    "resource",
    "sharedsubpage",
    "subpage",
    "url",
)
VLE_ACTIVITY_FEATURES = [
    f"log1p_vle_clicks_activity_{activity}_cutoff"
    for activity in VLE_ACTIVITY_TYPES
]
MODEL_CATEGORICAL_FEATURES = [
    "code_module",
    "code_presentation",
    "module_presentation",
    "highest_education",
]
CORE_MODEL_NUMERIC_FEATURES = [
    "num_of_prev_attempts",
    "studied_credits",
    "date_registration",
    "module_presentation_length",
    "registration_missing",
    "assessment_due_count",
    "assessment_submission_count_cutoff",
    "assessment_scored_count_cutoff",
    "assessment_score_missing_count_cutoff",
    "assessment_due_submission_count_cutoff",
    "assessment_completion_rate_cutoff",
    "assessment_score_mean_cutoff",
    "assessment_score_min_cutoff",
    "assessment_score_max_cutoff",
    "assessment_weighted_score_cutoff",
    "assessment_type_count_cutoff",
    "assessment_late_count_cutoff",
    "assessment_banked_count_cutoff",
    "assessment_days_since_last_submission",
    "assessment_has_submission",
    "vle_event_count_cutoff",
    "log1p_vle_total_clicks_cutoff",
    "vle_active_days_cutoff",
    "log1p_vle_resource_count_cutoff",
    "vle_activity_type_count_cutoff",
    "vle_days_since_last_activity",
    "vle_first_activity_day",
    "log1p_vle_clicks_last_7_days",
    "log1p_vle_clicks_previous_7_days",
    "vle_click_change_7d",
    "log1p_vle_clicks_last_28_days",
    "log1p_vle_clicks_previous_28_days",
    "vle_active_days_last_28_days",
    "vle_active_days_previous_28_days",
    "vle_click_change_28d",
    "vle_clicks_per_active_day",
    "vle_has_activity",
] + VLE_ACTIVITY_FEATURES
ENGINEERED_NUMERIC_FEATURES = [
    "days_observed_at_cutoff",
    "vle_active_day_rate_cutoff",
    "log1p_vle_clicks_per_active_day",
    "vle_recent_28_click_share",
    "vle_recent_7_click_share",
    "vle_active_day_change_28d",
    "assessment_missed_due_count",
    "assessment_on_time_rate_cutoff",
]
WEIGHTED_PROGRESS_FEATURES = [
    "assessment_submitted_weight_cutoff",
    "assessment_due_submitted_weight_cutoff",
    "assessment_due_weight_cutoff",
    "assessment_weighted_points_cutoff",
    "assessment_due_weighted_points_cutoff",
    "assessment_weight_completion_rate_cutoff",
    "assessment_due_performance_rate_cutoff",
    "assessment_failed_score_count_cutoff",
    "assessment_failed_score_rate_cutoff",
    "assessment_late_days_mean_cutoff",
]
MODULE_CODES = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG")
MODULE_SLOPE_BASES = (
    "assessment_weighted_score_cutoff",
    "assessment_completion_rate_cutoff",
    "assessment_days_since_last_submission",
    "log1p_vle_total_clicks_cutoff",
    "log1p_vle_clicks_last_28_days",
    "vle_active_days_last_28_days",
    "vle_days_since_last_activity",
)
MODULE_SLOPE_FEATURES = [
    f"module_{module}_x_{feature}"
    for module in MODULE_CODES
    for feature in MODULE_SLOPE_BASES
]
MODEL_NUMERIC_FEATURES = (
    CORE_MODEL_NUMERIC_FEATURES
    + ENGINEERED_NUMERIC_FEATURES
    + WEIGHTED_PROGRESS_FEATURES
    + MODULE_SLOPE_FEATURES
)
MODEL_INTERACTION_FEATURES = [
    "assessment_weighted_score_cutoff",
    "assessment_completion_rate_cutoff",
    "log1p_vle_total_clicks_cutoff",
    "log1p_vle_clicks_last_28_days",
    "vle_active_days_last_28_days",
    "vle_days_since_last_activity",
    "num_of_prev_attempts",
    "studied_credits",
]
PROHIBITED_MODEL_FEATURES = {
    "id_student",
    "final_result",
    "At_Risk",
    "Academic_Fail",
    "Performance_Level",
    "date_unregistration",
}


def validate_input_dir(input_dir: Path) -> Path:
    """Return a resolved input directory after checking all seven tables."""
    resolved = input_dir.resolve()
    missing = [name for name in REQUIRED_FILES if not (resolved / name).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing cleaned input files in {resolved}: {joined}. "
            "Run `python src/oulad_pipeline.py clean data/raw` first."
        )
    return resolved


def _attempt_tuples(frame: pd.DataFrame) -> list[tuple[str, str, int]]:
    return list(
        zip(
            frame["code_module"].astype(str),
            frame["code_presentation"].astype(str),
            frame["id_student"].astype(int),
        )
    )


def build_base_cohort(input_dir: Path, cutoff_day: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the learner-attempt cohort that is observable at ``cutoff_day``.

    Attempts registered after the cutoff and attempts already unregistered by
    the cutoff are excluded.  ``date_unregistration`` is used only for cohort
    eligibility and is removed before modeling; it is never a feature.
    """
    input_dir = validate_input_dir(input_dir)
    info = pd.read_csv(input_dir / "studentInfo.csv")
    registration = pd.read_csv(input_dir / "studentRegistration.csv")
    courses = pd.read_csv(input_dir / "courses.csv")

    base = info.merge(registration, on=ATTEMPT_KEY, how="left", validate="one_to_one")
    base = base.merge(
        courses,
        on=["code_module", "code_presentation"],
        how="left",
        validate="many_to_one",
    )
    if base[ATTEMPT_KEY].duplicated().any():
        raise ValueError("Base cohort is not unique at learner-attempt grain.")

    registered_after_cutoff = base["date_registration"].notna() & (
        base["date_registration"] > cutoff_day
    )
    unregistered_by_cutoff = base["date_unregistration"].notna() & (
        base["date_unregistration"] <= cutoff_day
    )
    academic_outcome = base["final_result"].isin(["Fail", "Pass", "Distinction"])
    withdrawn_outcome = base["final_result"].eq("Withdrawn")
    eligible = ~(registered_after_cutoff | unregistered_by_cutoff) & academic_outcome

    counts = {
        "all_attempts": int(len(base)),
        "excluded_registered_after_cutoff": int(registered_after_cutoff.sum()),
        "excluded_unregistered_by_cutoff": int(unregistered_by_cutoff.sum()),
        "excluded_withdrawn_outcome": int(withdrawn_outcome.sum()),
        "eligible_attempts": int(eligible.sum()),
    }
    cohort = base.loc[eligible].copy()
    cohort["registration_missing"] = cohort["date_registration"].isna().astype("int8")
    cohort["Academic_Fail"] = cohort["final_result"].map(
        {"Fail": 1, "Pass": 0, "Distinction": 0}
    )
    # Keep the legacy internal column while downstream files are migrated. Its
    # meaning is now strictly academic failure, never withdrawal.
    cohort["At_Risk"] = cohort["Academic_Fail"]
    if cohort["At_Risk"].isna().any():
        unknown = sorted(cohort.loc[cohort["At_Risk"].isna(), "final_result"].unique())
        raise ValueError(f"Unknown final_result values: {unknown}")
    cohort["Academic_Fail"] = cohort["Academic_Fail"].astype("int8")
    cohort["At_Risk"] = cohort["At_Risk"].astype("int8")
    cohort["actual_status"] = np.where(
        cohort["Academic_Fail"].eq(1), "Fail", "Pass/Distinction"
    )
    cohort["cutoff_day"] = int(cutoff_day)
    return cohort, counts


def audit_cutoffs(
    input_dir: Path,
    cutoffs: Iterable[int] = (14, 28, 42),
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Measure cohort and event coverage for candidate prediction cutoffs."""
    input_dir = validate_input_dir(input_dir)
    cutoff_values = sorted({int(value) for value in cutoffs})
    if not cutoff_values or cutoff_values[0] < 0:
        raise ValueError("Cutoffs must be non-negative integers.")

    vle_attempts: dict[int, set[tuple[str, str, int]]] = {
        cutoff: set() for cutoff in cutoff_values
    }
    vle_event_rows = {cutoff: 0 for cutoff in cutoff_values}
    for chunk in pd.read_csv(
        input_dir / "studentVle.csv",
        usecols=ATTEMPT_KEY + ["date"],
        chunksize=chunksize,
    ):
        for cutoff in cutoff_values:
            observed = chunk.loc[chunk["date"].le(cutoff), ATTEMPT_KEY]
            vle_event_rows[cutoff] += int(len(observed))
            vle_attempts[cutoff].update(_attempt_tuples(observed))

    assessments = pd.read_csv(input_dir / "assessments.csv")
    assessment_events = pd.read_csv(input_dir / "studentAssessment.csv").merge(
        assessments[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment",
        how="left",
        validate="many_to_one",
    )
    assessment_attempts: dict[int, set[tuple[str, str, int]]] = {}
    for cutoff in cutoff_values:
        observed = assessment_events.loc[
            assessment_events["date_submitted"].le(cutoff), ATTEMPT_KEY
        ]
        assessment_attempts[cutoff] = set(_attempt_tuples(observed))

    rows: list[dict[str, object]] = []
    for cutoff in cutoff_values:
        cohort, cohort_counts = build_base_cohort(input_dir, cutoff)
        attempt_keys = _attempt_tuples(cohort)
        cohort["has_vle_by_cutoff"] = [key in vle_attempts[cutoff] for key in attempt_keys]
        cohort["has_assessment_by_cutoff"] = [
            key in assessment_attempts[cutoff] for key in attempt_keys
        ]

        due_counts = (
            assessments.loc[assessments["date"].notna() & assessments["date"].le(cutoff)]
            .groupby(["code_module", "code_presentation"])["id_assessment"]
            .nunique()
        )
        cohort["due_assessment_count"] = pd.MultiIndex.from_frame(
            cohort[["code_module", "code_presentation"]]
        ).map(due_counts).fillna(0).astype(int)

        def add_row(scope: str, group: pd.DataFrame, module: str, presentation: str) -> None:
            attempts = len(group)
            rows.append(
                {
                    "cutoff_day": cutoff,
                    "scope": scope,
                    "code_module": module,
                    "code_presentation": presentation,
                    "eligible_attempts": int(attempts),
                    "at_risk_count": int(group["At_Risk"].sum()),
                    "at_risk_rate": float(group["At_Risk"].mean()) if attempts else np.nan,
                    "vle_attempt_coverage": float(group["has_vle_by_cutoff"].mean()) if attempts else np.nan,
                    "assessment_submission_coverage": float(
                        group["has_assessment_by_cutoff"].mean()
                    ) if attempts else np.nan,
                    "mean_due_assessment_count": float(group["due_assessment_count"].mean())
                    if attempts
                    else np.nan,
                    "mean_remaining_course_days": float(
                        (group["module_presentation_length"] - cutoff).mean()
                    ) if attempts else np.nan,
                    "all_attempts": cohort_counts["all_attempts"] if scope == "overall" else np.nan,
                    "excluded_registered_after_cutoff": cohort_counts[
                        "excluded_registered_after_cutoff"
                    ] if scope == "overall" else np.nan,
                    "excluded_unregistered_by_cutoff": cohort_counts[
                        "excluded_unregistered_by_cutoff"
                    ] if scope == "overall" else np.nan,
                    "vle_event_rows_observed": vle_event_rows[cutoff]
                    if scope == "overall"
                    else np.nan,
                }
            )

        add_row("overall", cohort, "ALL", "ALL")
        for (module, presentation), group in cohort.groupby(
            ["code_module", "code_presentation"], sort=True, observed=True
        ):
            add_row("module_presentation", group, str(module), str(presentation))

    return pd.DataFrame(rows)


def _assessment_features(input_dir: Path, cutoff_day: int) -> pd.DataFrame:
    assessments = pd.read_csv(input_dir / "assessments.csv")
    submissions = pd.read_csv(input_dir / "studentAssessment.csv")
    events = submissions.merge(
        assessments[
            [
                "id_assessment",
                "code_module",
                "code_presentation",
                "assessment_type",
                "date",
                "weight",
            ]
        ],
        on="id_assessment",
        how="left",
        validate="many_to_one",
    )
    events = events.loc[events["date_submitted"].le(cutoff_day)].copy()
    events["is_due_by_cutoff"] = events["date"].notna() & events["date"].le(cutoff_day)
    events["is_late"] = events["date"].notna() & events["date_submitted"].gt(events["date"])
    events["weighted_score_component"] = events["score"] * events["weight"]
    events["scored_weight"] = events["weight"].where(events["score"].notna())
    # These components retain how much scheduled assessment weight a learner
    # has actually completed.  The existing weighted mean deliberately removes
    # that information, so two learners can have the same mean despite one
    # having completed much less of the work due by the cutoff.
    events["submitted_weight"] = events["weight"].where(events["score"].notna())
    events["due_submitted_weight"] = events["weight"].where(
        events["is_due_by_cutoff"] & events["score"].notna()
    )
    events["due_weighted_score_component"] = events[
        "weighted_score_component"
    ].where(events["is_due_by_cutoff"])
    events["score_below_40"] = events["score"].notna() & events["score"].lt(40)
    events["late_days"] = (
        events["date_submitted"] - events["date"]
    ).where(events["date"].notna()).clip(lower=0)

    grouped = events.groupby(ATTEMPT_KEY, observed=True, dropna=False)
    features = grouped.agg(
        assessment_submission_count_cutoff=("id_assessment", "size"),
        assessment_scored_count_cutoff=("score", "count"),
        assessment_score_missing_count_cutoff=("score", lambda values: int(values.isna().sum())),
        assessment_due_submission_count_cutoff=("is_due_by_cutoff", "sum"),
        assessment_score_mean_cutoff=("score", "mean"),
        assessment_score_min_cutoff=("score", "min"),
        assessment_score_max_cutoff=("score", "max"),
        assessment_weighted_score_sum=("weighted_score_component", "sum"),
        assessment_scored_weight_sum=("scored_weight", "sum"),
        assessment_submitted_weight_cutoff=("submitted_weight", "sum"),
        assessment_due_submitted_weight_cutoff=("due_submitted_weight", "sum"),
        assessment_due_weighted_score_sum=("due_weighted_score_component", "sum"),
        assessment_failed_score_count_cutoff=("score_below_40", "sum"),
        assessment_late_days_mean_cutoff=("late_days", "mean"),
        assessment_late_count_cutoff=("is_late", "sum"),
        assessment_banked_count_cutoff=("is_banked", "sum"),
        assessment_type_count_cutoff=("assessment_type", "nunique"),
        assessment_last_submission_day=("date_submitted", "max"),
    ).reset_index()
    features["assessment_weighted_score_cutoff"] = np.where(
        features["assessment_scored_weight_sum"].gt(0),
        features["assessment_weighted_score_sum"] / features["assessment_scored_weight_sum"],
        np.nan,
    )
    features["assessment_weighted_points_cutoff"] = (
        features["assessment_weighted_score_sum"] / 100.0
    )
    features["assessment_due_weighted_points_cutoff"] = (
        features["assessment_due_weighted_score_sum"] / 100.0
    )
    return features.drop(columns=["assessment_weighted_score_sum", "assessment_scored_weight_sum"])


def _vle_features(
    input_dir: Path,
    cutoff_day: int,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    observed_chunks: list[pd.DataFrame] = []
    usecols = ATTEMPT_KEY + ["id_site", "date", "sum_click"]
    dtypes = {
        "code_module": "category",
        "code_presentation": "category",
        "id_student": "int32",
        "id_site": "int32",
        "date": "int16",
        "sum_click": "int32",
    }
    for chunk in pd.read_csv(
        input_dir / "studentVle.csv",
        usecols=usecols,
        dtype=dtypes,
        chunksize=chunksize,
    ):
        observed = chunk.loc[chunk["date"].le(cutoff_day)].copy()
        if not observed.empty:
            observed_chunks.append(observed)
    if not observed_chunks:
        return pd.DataFrame(columns=ATTEMPT_KEY)

    events = pd.concat(observed_chunks, ignore_index=True)
    activity_lookup = pd.read_csv(
        input_dir / "vle.csv", usecols=["id_site", "activity_type"]
    ).drop_duplicates("id_site")
    events = events.merge(activity_lookup, on="id_site", how="left", validate="many_to_one")
    if events["activity_type"].isna().any():
        raise ValueError("Some studentVle rows do not match vle.activity_type.")
    unexpected_activity_types = sorted(
        set(events["activity_type"].astype(str)) - set(VLE_ACTIVITY_TYPES)
    )
    if unexpected_activity_types:
        raise ValueError(f"Unknown VLE activity types: {unexpected_activity_types}")

    last_7_start = cutoff_day - 6
    previous_7_start = cutoff_day - 13
    last_28_start = cutoff_day - 27
    previous_28_start = cutoff_day - 55
    events["clicks_last_7"] = events["sum_click"].where(
        events["date"].between(last_7_start, cutoff_day), 0
    )
    events["clicks_previous_7"] = events["sum_click"].where(
        events["date"].between(previous_7_start, last_7_start - 1), 0
    )
    events["clicks_last_28"] = events["sum_click"].where(
        events["date"].between(last_28_start, cutoff_day), 0
    )
    events["clicks_previous_28"] = events["sum_click"].where(
        events["date"].between(previous_28_start, last_28_start - 1), 0
    )
    events["active_day_last_28"] = events["date"].where(
        events["date"].between(last_28_start, cutoff_day)
    )
    events["active_day_previous_28"] = events["date"].where(
        events["date"].between(previous_28_start, last_28_start - 1)
    )
    grouped = events.groupby(ATTEMPT_KEY, observed=True, dropna=False)
    summary = grouped.agg(
        vle_event_count_cutoff=("sum_click", "size"),
        vle_total_clicks_cutoff=("sum_click", "sum"),
        vle_active_days_cutoff=("date", "nunique"),
        vle_resource_count_cutoff=("id_site", "nunique"),
        vle_activity_type_count_cutoff=("activity_type", "nunique"),
        vle_first_activity_day=("date", "min"),
        vle_last_activity_day=("date", "max"),
        vle_clicks_last_7_days=("clicks_last_7", "sum"),
        vle_clicks_previous_7_days=("clicks_previous_7", "sum"),
        vle_clicks_last_28_days=("clicks_last_28", "sum"),
        vle_clicks_previous_28_days=("clicks_previous_28", "sum"),
        vle_active_days_last_28_days=("active_day_last_28", "nunique"),
        vle_active_days_previous_28_days=("active_day_previous_28", "nunique"),
    ).reset_index()

    activity_clicks = (
        events.groupby(ATTEMPT_KEY + ["activity_type"], observed=True, dropna=False)[
            "sum_click"
        ]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=VLE_ACTIVITY_TYPES, fill_value=0)
    )
    activity_clicks.columns = VLE_ACTIVITY_FEATURES
    activity_clicks = np.log1p(activity_clicks.clip(lower=0)).reset_index()
    return summary.merge(activity_clicks, on=ATTEMPT_KEY, how="left", validate="one_to_one")


def build_feature_snapshot(
    input_dir: Path,
    cutoff_day: int,
    chunksize: int = 500_000,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return one leakage-safe row per eligible learner-attempt."""
    input_dir = validate_input_dir(input_dir)
    cohort, cohort_counts = build_base_cohort(input_dir, cutoff_day)
    assessment = _assessment_features(input_dir, cutoff_day)
    vle = _vle_features(input_dir, cutoff_day, chunksize=chunksize)

    due_schedule = (
        pd.read_csv(input_dir / "assessments.csv")
        .loc[lambda frame: frame["date"].notna() & frame["date"].le(cutoff_day)]
        .groupby(["code_module", "code_presentation"], observed=True)
        .agg(
            assessment_due_count=("id_assessment", "nunique"),
            assessment_due_weight_cutoff=("weight", "sum"),
        )
        .reset_index()
    )
    snapshot = cohort.merge(assessment, on=ATTEMPT_KEY, how="left", validate="one_to_one")
    snapshot = snapshot.merge(vle, on=ATTEMPT_KEY, how="left", validate="one_to_one")
    snapshot = snapshot.merge(
        due_schedule,
        on=["code_module", "code_presentation"],
        how="left",
        validate="many_to_one",
    )
    snapshot["module_presentation"] = (
        snapshot["code_module"].astype(str)
        + "_"
        + snapshot["code_presentation"].astype(str)
    )

    count_columns = [
        "assessment_due_count",
        "assessment_submission_count_cutoff",
        "assessment_scored_count_cutoff",
        "assessment_score_missing_count_cutoff",
        "assessment_due_submission_count_cutoff",
        "assessment_late_count_cutoff",
        "assessment_banked_count_cutoff",
        "assessment_type_count_cutoff",
        "assessment_submitted_weight_cutoff",
        "assessment_due_submitted_weight_cutoff",
        "assessment_weighted_points_cutoff",
        "assessment_due_weighted_points_cutoff",
        "assessment_failed_score_count_cutoff",
        "vle_event_count_cutoff",
        "vle_total_clicks_cutoff",
        "vle_active_days_cutoff",
        "vle_resource_count_cutoff",
        "vle_activity_type_count_cutoff",
        "vle_clicks_last_7_days",
        "vle_clicks_previous_7_days",
        "vle_clicks_last_28_days",
        "vle_clicks_previous_28_days",
        "vle_active_days_last_28_days",
        "vle_active_days_previous_28_days",
    ]
    count_columns += VLE_ACTIVITY_FEATURES
    snapshot[count_columns] = snapshot[count_columns].fillna(0)
    snapshot["assessment_has_submission"] = snapshot[
        "assessment_submission_count_cutoff"
    ].gt(0).astype("int8")
    snapshot["assessment_completion_rate_cutoff"] = np.where(
        snapshot["assessment_due_count"].gt(0),
        snapshot["assessment_due_submission_count_cutoff"]
        / snapshot["assessment_due_count"],
        np.nan,
    )
    snapshot["assessment_weight_completion_rate_cutoff"] = np.where(
        snapshot["assessment_due_weight_cutoff"].gt(0),
        snapshot["assessment_due_submitted_weight_cutoff"]
        / snapshot["assessment_due_weight_cutoff"],
        np.nan,
    )
    snapshot["assessment_due_performance_rate_cutoff"] = np.where(
        snapshot["assessment_due_weight_cutoff"].gt(0),
        snapshot["assessment_due_weighted_points_cutoff"]
        / snapshot["assessment_due_weight_cutoff"],
        np.nan,
    )
    snapshot["assessment_failed_score_rate_cutoff"] = np.where(
        snapshot["assessment_scored_count_cutoff"].gt(0),
        snapshot["assessment_failed_score_count_cutoff"]
        / snapshot["assessment_scored_count_cutoff"],
        np.nan,
    )
    registration_start = snapshot["date_registration"].fillna(0).clip(lower=0)
    registration_start = np.minimum(registration_start, float(cutoff_day))
    snapshot["days_observed_at_cutoff"] = (
        cutoff_day - registration_start + 1
    ).clip(lower=1)
    snapshot["vle_active_day_rate_cutoff"] = (
        snapshot["vle_active_days_cutoff"] / snapshot["days_observed_at_cutoff"]
    ).clip(0, 1)
    snapshot["vle_recent_28_click_share"] = np.where(
        snapshot["vle_total_clicks_cutoff"].gt(0),
        snapshot["vle_clicks_last_28_days"] / snapshot["vle_total_clicks_cutoff"],
        0.0,
    )
    snapshot["vle_recent_7_click_share"] = np.where(
        snapshot["vle_clicks_last_28_days"].gt(0),
        snapshot["vle_clicks_last_7_days"]
        / snapshot["vle_clicks_last_28_days"],
        0.0,
    )
    snapshot["vle_active_day_change_28d"] = (
        snapshot["vle_active_days_last_28_days"]
        - snapshot["vle_active_days_previous_28_days"]
    ) / 28.0
    snapshot["assessment_missed_due_count"] = (
        snapshot["assessment_due_count"]
        - snapshot["assessment_due_submission_count_cutoff"]
    ).clip(lower=0)
    snapshot["assessment_on_time_rate_cutoff"] = np.where(
        snapshot["assessment_submission_count_cutoff"].gt(0),
        1.0
        - snapshot["assessment_late_count_cutoff"]
        / snapshot["assessment_submission_count_cutoff"],
        np.nan,
    )
    snapshot["assessment_days_since_last_submission"] = (
        cutoff_day - snapshot["assessment_last_submission_day"]
    )
    snapshot["vle_has_activity"] = snapshot["vle_event_count_cutoff"].gt(0).astype("int8")
    snapshot["vle_days_since_last_activity"] = cutoff_day - snapshot["vle_last_activity_day"]
    snapshot["vle_clicks_per_active_day"] = np.where(
        snapshot["vle_active_days_cutoff"].gt(0),
        snapshot["vle_total_clicks_cutoff"] / snapshot["vle_active_days_cutoff"],
        0.0,
    )
    snapshot["vle_click_change_7d"] = (
        snapshot["vle_clicks_last_7_days"] - snapshot["vle_clicks_previous_7_days"]
    )
    snapshot["vle_click_change_28d"] = (
        snapshot["vle_clicks_last_28_days"]
        - snapshot["vle_clicks_previous_28_days"]
    )
    for source, target in (
        ("vle_total_clicks_cutoff", "log1p_vle_total_clicks_cutoff"),
        ("vle_resource_count_cutoff", "log1p_vle_resource_count_cutoff"),
        ("vle_clicks_last_7_days", "log1p_vle_clicks_last_7_days"),
        ("vle_clicks_previous_7_days", "log1p_vle_clicks_previous_7_days"),
        ("vle_clicks_last_28_days", "log1p_vle_clicks_last_28_days"),
        ("vle_clicks_previous_28_days", "log1p_vle_clicks_previous_28_days"),
    ):
        snapshot[target] = np.log1p(snapshot[source].clip(lower=0))
    snapshot["log1p_vle_clicks_per_active_day"] = np.log1p(
        snapshot["vle_clicks_per_active_day"].clip(lower=0)
    )

    module_slopes = {
        f"module_{module}_x_{feature}": np.where(
            snapshot["code_module"].eq(module), snapshot[feature], 0.0
        )
        for module in MODULE_CODES
        for feature in MODULE_SLOPE_BASES
    }
    snapshot = pd.concat(
        [snapshot, pd.DataFrame(module_slopes, index=snapshot.index)], axis=1
    )

    if snapshot[ATTEMPT_KEY].duplicated().any():
        raise ValueError("Feature snapshot is not unique at learner-attempt grain.")
    missing_features = [
        feature
        for feature in (
            MODEL_CATEGORICAL_FEATURES
            + MODEL_NUMERIC_FEATURES
            + MODEL_INTERACTION_FEATURES
        )
        if feature not in snapshot.columns
    ]
    if missing_features:
        raise ValueError(f"Feature snapshot is missing columns: {missing_features}")
    if PROHIBITED_MODEL_FEATURES.intersection(
        MODEL_CATEGORICAL_FEATURES + MODEL_NUMERIC_FEATURES
    ):
        raise AssertionError("A prohibited column entered the model feature contract.")
    return snapshot, cohort_counts
