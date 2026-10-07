"""Build compact, reproducible data marts for the Streamlit dashboard.

The dashboard must stay responsive and must not join the 8.4-million-row VLE
event table during a user request.  This script performs those joins once and
writes three compressed, documented tables at the grains needed by the two
dashboard pages.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "dashboard"

ATTEMPT_KEY = ["code_module", "code_presentation", "id_student"]
FILTER_COLUMNS = ["code_module", "code_presentation", "gender", "region"]


def _student_lookup() -> pd.DataFrame:
    columns = ATTEMPT_KEY + [
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "num_of_prev_attempts",
        "final_result",
    ]
    frame = pd.read_csv(INTERIM_DIR / "studentInfo.csv", usecols=columns)
    if frame.duplicated(ATTEMPT_KEY).any():
        raise ValueError("studentInfo.csv is not unique at learning-attempt grain")
    frame["Academic_Fail"] = frame["final_result"].eq("Fail").astype(int)
    frame["Withdrawn_Flag"] = frame["final_result"].eq("Withdrawn").astype(int)
    # Legacy chart code reads At_Risk; its canonical meaning is now Fail only.
    frame["At_Risk"] = frame["Academic_Fail"]
    return frame


def build_assessment_marts(student: pd.DataFrame) -> None:
    assessments = pd.read_csv(INTERIM_DIR / "assessments.csv").rename(
        columns={"date": "due_date"}
    )
    if assessments.duplicated("id_assessment").any():
        raise ValueError("assessments.csv has duplicate id_assessment values")

    submissions = pd.read_csv(INTERIM_DIR / "studentAssessment.csv")
    submissions = submissions.merge(
        assessments[
            [
                "id_assessment",
                "code_module",
                "code_presentation",
                "assessment_type",
                "due_date",
                "weight",
            ]
        ],
        on="id_assessment",
        how="left",
        validate="many_to_one",
    )
    submissions = submissions.merge(
        student[
            ATTEMPT_KEY
            + [
                "gender",
                "region",
                "num_of_prev_attempts",
                "final_result",
                "Academic_Fail",
                "Withdrawn_Flag",
                "At_Risk",
            ]
        ],
        on=ATTEMPT_KEY,
        how="inner",
        validate="many_to_one",
    )
    submissions["submission_delay"] = (
        submissions["date_submitted"] - submissions["due_date"]
    )
    submissions = submissions.dropna(
        subset=["score", "date_submitted", "due_date", "submission_delay"]
    ).copy()
    submissions["score"] = pd.to_numeric(submissions["score"], errors="raise")
    submissions["submission_delay"] = submissions["submission_delay"].astype(int)
    submissions.to_csv(
        OUTPUT_DIR / "assessment_submissions.csv.gz",
        index=False,
        compression="gzip",
    )

    assessments[
        [
            "code_module",
            "code_presentation",
            "id_assessment",
            "assessment_type",
            "due_date",
            "weight",
        ]
    ].to_csv(OUTPUT_DIR / "assessment_deadlines.csv", index=False)


def build_vle_marts(student: pd.DataFrame, chunksize: int = 500_000) -> None:
    vle_lookup = pd.read_csv(
        INTERIM_DIR / "vle.csv",
        usecols=["id_site", "code_module", "code_presentation", "activity_type"],
    )
    vle_key = ["id_site", "code_module", "code_presentation"]
    if vle_lookup.duplicated(vle_key).any():
        raise ValueError("vle.csv is not unique at resource grain")

    student_dimensions = student[
        ATTEMPT_KEY
        + ["gender", "region", "final_result", "Academic_Fail", "Withdrawn_Flag", "At_Risk"]
    ]
    daily_parts: list[pd.DataFrame] = []
    activity_parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        INTERIM_DIR / "studentVle.csv",
        usecols=ATTEMPT_KEY + ["id_site", "date", "sum_click"],
        chunksize=chunksize,
    ):
        enriched = chunk.merge(
            student_dimensions,
            on=ATTEMPT_KEY,
            how="inner",
            validate="many_to_one",
        ).merge(
            vle_lookup,
            on=["id_site", "code_module", "code_presentation"],
            how="left",
            validate="many_to_one",
        )
        enriched["activity_type"] = enriched["activity_type"].fillna("unknown")

        daily_parts.append(
            enriched.groupby(
                FILTER_COLUMNS + ["final_result", "Academic_Fail", "At_Risk", "date"],
                observed=True,
                as_index=False,
            )["sum_click"].sum()
        )
        activity_parts.append(
            enriched.groupby(
                FILTER_COLUMNS + ["final_result", "Academic_Fail", "At_Risk", "activity_type"],
                observed=True,
                as_index=False,
            )["sum_click"].sum()
        )

    daily = pd.concat(daily_parts, ignore_index=True)
    daily = daily.groupby(
        FILTER_COLUMNS + ["final_result", "Academic_Fail", "At_Risk", "date"],
        observed=True,
        as_index=False,
    )["sum_click"].sum()
    daily.to_csv(
        OUTPUT_DIR / "vle_daily_profile.csv.gz", index=False, compression="gzip"
    )

    activity = pd.concat(activity_parts, ignore_index=True)
    activity = activity.groupby(
        FILTER_COLUMNS + ["final_result", "Academic_Fail", "At_Risk", "activity_type"],
        observed=True,
        as_index=False,
    )["sum_click"].sum()
    activity.to_csv(
        OUTPUT_DIR / "vle_activity_summary.csv.gz", index=False, compression="gzip"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    student = _student_lookup()
    build_assessment_marts(student)
    build_vle_marts(student)
    print(f"Dashboard marts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
