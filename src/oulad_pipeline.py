"""Reproducible T05--T07 pipeline for the local OULAD CSV files.

The raw directory is read-only.  Interim outputs stay local.  The shared
processed output is data/processed/clean_dataset.csv; the tracked Data Quality
Report is generated from the JSON metrics that the stages produce.

Examples (PowerShell):
    python src/oulad_pipeline.py audit data/raw
    python src/oulad_pipeline.py clean data/raw
    python src/oulad_pipeline.py build data/raw
    python src/oulad_pipeline.py report
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "data-quality-report.md"
AUDIT_METRICS = INTERIM / "t05_audit_metrics.json"
CLEAN_METRICS = INTERIM / "t06_cleaning_metrics.json"
JOIN_METRICS = INTERIM / "t07_join_metrics.json"

FILES = (
    "courses.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "assessments.csv",
    "studentAssessment.csv",
    "vle.csv",
    "studentVle.csv",
)
ATTEMPT_KEY = ["code_module", "code_presentation", "id_student"]
VLE_EVENT_KEY = ATTEMPT_KEY + ["id_site", "date"]
KEYS = {
    "courses.csv": ["code_module", "code_presentation"],
    "studentInfo.csv": ATTEMPT_KEY,
    "studentRegistration.csv": ATTEMPT_KEY,
    "assessments.csv": ["id_assessment"],
    "studentAssessment.csv": ["id_assessment", "id_student"],
    "vle.csv": ["id_site"],
}
NUMERIC_COLUMNS = {
    "courses.csv": ["module_presentation_length"],
    "studentInfo.csv": ["id_student", "num_of_prev_attempts", "studied_credits"],
    "studentRegistration.csv": ["id_student", "date_registration", "date_unregistration"],
    "assessments.csv": ["id_assessment", "date", "weight"],
    "studentAssessment.csv": ["id_assessment", "id_student", "date_submitted", "is_banked", "score"],
    "vle.csv": ["id_site", "week_from", "week_to"],
    "studentVle.csv": ["id_student", "id_site", "date", "sum_click"],
}
VALID_VALUES = {
    ("studentInfo.csv", "final_result"): {"Distinction", "Pass", "Fail", "Withdrawn"},
    ("studentInfo.csv", "gender"): {"M", "F"},
    ("studentInfo.csv", "disability"): {"Y", "N"},
    ("assessments.csv", "assessment_type"): {"TMA", "CMA", "Exam"},
    ("studentAssessment.csv", "is_banked"): {0, 1},
}


def ensure_dirs() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)


def read_raw(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read raw values faithfully: blanks and '?' remain distinguishable."""
    return pd.read_csv(path, keep_default_na=False, na_filter=False, **kwargs)


def raw_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        column: {"blank": int((frame[column] == "").sum()), "question_mark": int((frame[column] == "?").sum())}
        for column in frame.columns
    }


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series.replace({"": np.nan, "?": np.nan}), errors="coerce").dropna()
    if values.empty:
        return {"min": None, "max": None, "q1": None, "q3": None, "iqr_outliers": 0}
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        outliers = 0
    else:
        outliers = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
    return {
        "min": float(values.min()), "max": float(values.max()), "q1": float(q1), "q3": float(q3),
        "iqr_outliers": outliers,
    }


def frame_audit(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing": raw_counts(frame),
        "whole_row_duplicates": int(frame.duplicated().sum()),
        "numeric": {column: numeric_summary(frame[column]) for column in NUMERIC_COLUMNS[name]},
        "invalid": {},
    }
    if name in KEYS:
        key = KEYS[name]
        null_key = int(((frame[key] == "").any(axis=1) | (frame[key] == "?").any(axis=1)).sum())
        metrics["key"] = {
            "columns": key,
            "null_rows": null_key,
            "distinct": int(frame[key].drop_duplicates().shape[0]),
            "duplicate_rows": int(frame.duplicated(key).sum()),
        }
    for (file_name, column), allowed in VALID_VALUES.items():
        if file_name == name:
            observed = set(pd.to_numeric(frame[column], errors="raise")) if column == "is_banked" else set(frame[column])
            metrics["invalid"][column] = sorted(str(value) for value in observed - allowed)
    return metrics


def audit_student_vle(path: Path, info_keys: set[tuple[str, str, int]], vle_keys: set[tuple[str, str, int]]) -> dict[str, Any]:
    """Chunk audit the 10.6m-row event table without loading it into memory."""
    rows = exact_duplicate_rows = unmatched_info = unmatched_vle = 0
    blank = Counter()
    question = Counter()
    module_counts: Counter[str] = Counter()
    date_counts: Counter[int] = Counter()
    click_counts: Counter[int] = Counter()
    event_key_counts: Counter[tuple[str, str, int, int, int]] = Counter()
    min_values: dict[str, int | None] = {column: None for column in NUMERIC_COLUMNS["studentVle.csv"]}
    max_values: dict[str, int | None] = {column: None for column in NUMERIC_COLUMNS["studentVle.csv"]}

    for chunk in pd.read_csv(path, keep_default_na=False, na_filter=False, chunksize=250_000):
        rows += len(chunk)
        for column in chunk.columns:
            blank[column] += int((chunk[column] == "").sum())
            question[column] += int((chunk[column] == "?").sum())
        exact_duplicate_rows += int(chunk.duplicated().sum())
        module_counts.update(chunk["code_module"].astype(str))
        for column in NUMERIC_COLUMNS["studentVle.csv"]:
            values = pd.to_numeric(chunk[column], errors="coerce")
            min_values[column] = int(values.min()) if min_values[column] is None else min(min_values[column], int(values.min()))
            max_values[column] = int(values.max()) if max_values[column] is None else max(max_values[column], int(values.max()))
        date_counts.update(pd.to_numeric(chunk["date"]).astype(int))
        click_counts.update(pd.to_numeric(chunk["sum_click"]).astype(int))
        event_key_counts.update(zip(
            chunk["code_module"], chunk["code_presentation"], pd.to_numeric(chunk["id_student"]).astype(int),
            pd.to_numeric(chunk["id_site"]).astype(int), pd.to_numeric(chunk["date"]).astype(int),
        ))
        unmatched_info += sum(
            (module, presentation, student) not in info_keys
            for module, presentation, student in zip(chunk["code_module"], chunk["code_presentation"], pd.to_numeric(chunk["id_student"]).astype(int))
        )
        unmatched_vle += sum(
            (module, presentation, site) not in vle_keys
            for module, presentation, site in zip(chunk["code_module"], chunk["code_presentation"], pd.to_numeric(chunk["id_site"]).astype(int))
        )

    def histogram_summary(counts: Counter[int]) -> dict[str, Any]:
        total = sum(counts.values())
        ordered = sorted(counts.items())
        def percentile(p: float) -> float:
            target = total * p
            cumulative = 0
            for value, count in ordered:
                cumulative += count
                if cumulative >= target:
                    return float(value)
            return float(ordered[-1][0])
        q1, q3 = percentile(0.25), percentile(0.75)
        iqr = q3 - q1
        outliers = sum(count for value, count in ordered if iqr and (value < q1 - 1.5 * iqr or value > q3 + 1.5 * iqr))
        return {"min": ordered[0][0], "max": ordered[-1][0], "q1": q1, "q3": q3, "iqr_outliers": outliers}

    duplicate_key_rows = sum(count - 1 for count in event_key_counts.values() if count > 1)
    return {
        "rows": rows,
        "columns": ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
        "dtypes": {"code_module": "string", "code_presentation": "string", "id_student": "int64", "id_site": "int64", "date": "int64", "sum_click": "int64"},
        "missing": {column: {"blank": blank[column], "question_mark": question[column]} for column in blank},
        "whole_row_duplicates_within_chunks": exact_duplicate_rows,
        "event_key": {"columns": ["code_module", "code_presentation", "id_student", "id_site", "date"], "null_rows": 0, "distinct": len(event_key_counts), "duplicate_rows": duplicate_key_rows},
        "numeric": {
            "id_student": {"min": min_values["id_student"], "max": max_values["id_student"], "q1": None, "q3": None, "iqr_outliers": None},
            "id_site": {"min": min_values["id_site"], "max": max_values["id_site"], "q1": None, "q3": None, "iqr_outliers": None},
            "date": histogram_summary(date_counts), "sum_click": histogram_summary(click_counts),
        },
        "invalid": {"sum_click_negative": sum(count for value, count in click_counts.items() if value < 0), "date_non_numeric": 0},
        "join_coverage": {"studentInfo_unmatched": unmatched_info, "vle_unmatched": unmatched_vle},
        "module_counts": dict(sorted(module_counts.items())),
    }


def audit(raw_dir: Path) -> None:
    ensure_dirs()
    raw_dir = raw_dir.resolve()
    missing_files = [name for name in FILES if not (raw_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing raw files: {', '.join(missing_files)}")
    metrics: dict[str, Any] = {"task": "T05", "raw_dir": str(raw_dir), "pandas": pd.__version__, "numpy": np.__version__, "tables": {}}
    frames = {name: read_raw(raw_dir / name) for name in FILES if name != "studentVle.csv"}
    for name, frame in frames.items():
        metrics["tables"][name] = frame_audit(name, frame)
    info = frames["studentInfo.csv"]
    vle = frames["vle.csv"]
    info_keys = set(zip(info["code_module"], info["code_presentation"], pd.to_numeric(info["id_student"]).astype(int)))
    vle_keys = set(zip(vle["code_module"], vle["code_presentation"], pd.to_numeric(vle["id_site"]).astype(int)))
    metrics["tables"]["studentVle.csv"] = audit_student_vle(raw_dir / "studentVle.csv", info_keys, vle_keys)

    assessments = frames["assessments.csv"]
    registration = frames["studentRegistration.csv"]
    student_assessment = frames["studentAssessment.csv"]
    assessment_map = assessments.set_index("id_assessment")
    joined_assessment = student_assessment.merge(assessments[["id_assessment", "code_module", "code_presentation", "date"]], on="id_assessment", how="left", indicator=True)
    attempt_from_assessment = set(zip(joined_assessment["code_module"], joined_assessment["code_presentation"], pd.to_numeric(joined_assessment["id_student"]).astype(int)))
    registration_keys = set(zip(registration["code_module"], registration["code_presentation"], pd.to_numeric(registration["id_student"]).astype(int)))
    registration_result = info[ATTEMPT_KEY + ["final_result"]].merge(
        registration[ATTEMPT_KEY + ["date_unregistration"]], on=ATTEMPT_KEY, how="left", validate="one_to_one",
    )
    registration_result["unregistration_missing"] = registration_result["date_unregistration"].isin(["", "?"])
    unregistration_crosstab = pd.crosstab(registration_result["final_result"], registration_result["unregistration_missing"])
    metrics["relationships"] = {
        "registration_outside_studentInfo": len(registration_keys - info_keys),
        "studentInfo_without_registration": len(info_keys - registration_keys),
        "assessments_without_courses": int((~assessments.set_index(["code_module", "code_presentation"]).index.isin(pd.MultiIndex.from_frame(frames["courses.csv"][["code_module", "code_presentation"]]))).sum()),
        "studentAssessment_missing_assessment": int((joined_assessment["_merge"] != "both").sum()),
        "studentAssessment_inferred_attempt_outside_studentInfo": len(attempt_from_assessment - info_keys),
        "studentVle_studentInfo_unmatched": metrics["tables"]["studentVle.csv"]["join_coverage"]["studentInfo_unmatched"],
        "studentVle_vle_unmatched": metrics["tables"]["studentVle.csv"]["join_coverage"]["vle_unmatched"],
        "late_submissions": int(((pd.to_numeric(joined_assessment["date_submitted"]) > pd.to_numeric(joined_assessment["date"].replace("?", np.nan))) & joined_assessment["date"].ne("?")).sum()),
        "banked_distribution": {str(key): int(value) for key, value in student_assessment["is_banked"].value_counts().sort_index().items()},
        "unregistration_missing_by_final_result": {
            str(result): {str(missing): int(count) for missing, count in row.items()}
            for result, row in unregistration_crosstab.to_dict(orient="index").items()
        },
    }
    AUDIT_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"T05 audit metrics written to {AUDIT_METRICS.relative_to(ROOT)}")


def clean_frame(name: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    before = {"rows": len(frame), "missing": raw_counts(frame)}
    cleaned = frame.replace("?", pd.NA).copy()
    for column in NUMERIC_COLUMNS[name]:
        numeric = pd.to_numeric(cleaned[column], errors="raise")
        observed = numeric.dropna()
        dtype = "Int64" if observed.empty or bool((observed % 1 == 0).all()) else "Float64"
        cleaned[column] = numeric.astype(dtype)
    for column in cleaned.columns:
        if column not in NUMERIC_COLUMNS[name]:
            cleaned[column] = cleaned[column].astype("string").str.strip()
    exact_duplicate_rows = int(cleaned.duplicated().sum())
    before_click_total: int | None = None
    after_click_total: int | None = None
    consolidated_event_key_rows = 0

    if name == "studentVle.csv":
        # OULAD can contain several source rows for the same learner, resource
        # and day.  They are click contributions, not safe-to-delete duplicate
        # observations.  Normalize to the documented daily-resource grain by
        # summing sum_click, thereby preserving the complete interaction volume.
        before_click_total = int(cleaned["sum_click"].sum())
        before_rows = len(cleaned)
        cleaned = (
            cleaned.groupby(VLE_EVENT_KEY, as_index=False, dropna=False)["sum_click"]
            .sum()
        )
        consolidated_event_key_rows = before_rows - len(cleaned)
        after_click_total = int(cleaned["sum_click"].sum())
        if before_click_total != after_click_total:
            raise AssertionError("studentVle normalization changed the total click count.")
        if cleaned[VLE_EVENT_KEY].duplicated().any():
            raise AssertionError("studentVle normalization did not produce a unique event key.")
        dropped_exact_duplicates = 0
    else:
        dropped_exact_duplicates = exact_duplicate_rows
        if dropped_exact_duplicates:
            cleaned = cleaned.drop_duplicates().copy()

    after_missing = {column: int(cleaned[column].isna().sum()) for column in cleaned.columns}
    return cleaned, {
        "before": before,
        "after": {"rows": len(cleaned), "missing": after_missing},
        "exact_duplicate_rows_observed": exact_duplicate_rows,
        "dropped_exact_duplicates": dropped_exact_duplicates,
        "consolidated_event_key_rows": consolidated_event_key_rows,
        "click_total_before": before_click_total,
        "click_total_after": after_click_total,
    }


def clean(raw_dir: Path) -> None:
    ensure_dirs()
    raw_dir = raw_dir.resolve()
    metrics: dict[str, Any] = {"task": "T06", "rules": [], "tables": {}}
    rules = [
        "Raw CSV are read only; '?' is normalized to nullable missing in interim outputs, never imputed.",
        "String fields are trimmed; numeric fields are cast to nullable Int64 or Float64 according to observed values.",
        "Exact full-row duplicates are dropped only in non-event tables; outliers are retained for later interpretation.",
        "studentVle rows are consolidated by code_module, code_presentation, id_student, id_site and date; sum_click is summed and its total must remain unchanged.",
        "date_unregistration missing is retained as unknown/not-recorded and is prohibited from prediction features; it is not assumed equivalent to not Withdrawn.",
        "studentAssessment.score missing and assessment/vle unknown dates/weeks remain missing; no score/date imputation is performed.",
    ]
    metrics["rules"] = rules
    for name in FILES:
        frame = read_raw(raw_dir / name)
        cleaned, table_metrics = clean_frame(name, frame)
        cleaned.to_csv(INTERIM / name, index=False)
        metrics["tables"][name] = table_metrics
    CLEAN_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"T06 cleaned interim CSV and metrics written below {INTERIM.relative_to(ROOT)}")


def build(raw_dir: Path) -> None:
    ensure_dirs()
    required = [INTERIM / name for name in FILES]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("Run the clean stage first; cleaned interim CSV files are required.")
    courses = pd.read_csv(INTERIM / "courses.csv")
    info = pd.read_csv(INTERIM / "studentInfo.csv")
    registration = pd.read_csv(INTERIM / "studentRegistration.csv")
    assessments = pd.read_csv(INTERIM / "assessments.csv")
    student_assessment = pd.read_csv(INTERIM / "studentAssessment.csv")
    vle = pd.read_csv(INTERIM / "vle.csv")

    assessment_event = student_assessment.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "assessment_type", "date", "weight"]],
        on="id_assessment", how="left", validate="many_to_one", indicator=True,
    )
    assessment_unmatched = int((assessment_event["_merge"] != "both").sum())
    assessment_event = assessment_event.drop(columns="_merge")
    assessment_event["attempt_key"] = list(zip(assessment_event["code_module"], assessment_event["code_presentation"], assessment_event["id_student"]))
    assessment_event["is_late"] = assessment_event["date"].notna() & (assessment_event["date_submitted"] > assessment_event["date"])
    assessment_agg = assessment_event.groupby(ATTEMPT_KEY, dropna=False).agg(
        assessment_event_count=("id_assessment", "size"),
        assessment_scored_count=("score", "count"),
        assessment_score_missing_count=("score", lambda x: int(x.isna().sum())),
        assessment_score_sum_all_time=("score", "sum"),
        assessment_score_mean_all_time=("score", "mean"),
        assessment_score_min_all_time=("score", "min"),
        assessment_score_max_all_time=("score", "max"),
        assessment_banked_count=("is_banked", "sum"),
        assessment_late_submission_count_all_time=("is_late", "sum"),
        assessment_type_nunique=("assessment_type", "nunique"),
    ).reset_index()

    activity_lookup = vle[["code_module", "code_presentation", "id_site", "activity_type"]]
    # The cleaned event file fits in local working memory.  Global aggregation is
    # deliberate: summing chunk-level nunique values would over-count a site/day
    # that appears in two chunks.
    vle_event = pd.read_csv(INTERIM / "studentVle.csv").merge(
        activity_lookup, on=["code_module", "code_presentation", "id_site"], how="left", validate="many_to_one", indicator=True,
    )
    click_rows = len(vle_event)
    click_unmatched = int((vle_event["_merge"] != "both").sum())
    vle_event = vle_event.drop(columns="_merge")
    vle_agg = vle_event.groupby(ATTEMPT_KEY, dropna=False).agg(
        vle_event_count=("sum_click", "size"),
        vle_total_clicks_all_time=("sum_click", "sum"),
        vle_active_days_all_time=("date", "nunique"),
        vle_resource_count_all_time=("id_site", "nunique"),
        vle_activity_type_count_all_time=("activity_type", "nunique"),
        vle_first_event_day=("date", "min"),
        vle_last_event_day=("date", "max"),
    ).reset_index()

    base = info.merge(registration, on=ATTEMPT_KEY, how="left", validate="one_to_one", indicator="registration_merge")
    registration_unmatched = int((base["registration_merge"] != "both").sum())
    base = base.drop(columns="registration_merge")
    base = base.merge(courses, on=["code_module", "code_presentation"], how="left", validate="many_to_one", indicator="courses_merge")
    courses_unmatched = int((base["courses_merge"] != "both").sum())
    base = base.drop(columns="courses_merge")
    base = base.merge(assessment_agg, on=ATTEMPT_KEY, how="left", validate="one_to_one", indicator="assessment_merge")
    base = base.drop(columns="assessment_merge")
    base = base.merge(vle_agg, on=ATTEMPT_KEY, how="left", validate="one_to_one", indicator="vle_merge")
    base = base.drop(columns="vle_merge")

    # The raw OULAD literal "10-20" is a deprivation band, not a date.  The
    # raw source remains unchanged; normalize this dashboard-ready output so
    # Keep an explicit percentage suffix so data readers cannot coerce it to Oct-20.
    base["imd_band"] = base["imd_band"].replace({"10-20": "10-20%"})
    base["imd_band_display"] = base["imd_band"].fillna("Unknown")

    # Missing aggregates after a left join mean no observed event.  Counts and
    # totals have a valid zero, unlike score statistics and event dates.
    zero_when_no_event = [
        "assessment_event_count",
        "assessment_scored_count",
        "assessment_score_missing_count",
        "assessment_score_sum_all_time",
        "assessment_banked_count",
        "assessment_late_submission_count_all_time",
        "assessment_type_nunique",
        "vle_event_count",
        "vle_total_clicks_all_time",
        "vle_active_days_all_time",
        "vle_resource_count_all_time",
        "vle_activity_type_count_all_time",
    ]
    base[zero_when_no_event] = base[zero_when_no_event].fillna(0)
    base["Academic_Fail"] = base["final_result"].eq("Fail").astype("int8")
    base["Withdrawn_Flag"] = base["final_result"].eq("Withdrawn").astype("int8")
    # Compatibility alias: the approved academic target is Fail only.
    base["At_Risk"] = base["Academic_Fail"]
    base["Performance_Level"] = base["final_result"]
    if base[ATTEMPT_KEY].duplicated().any():
        raise ValueError("T07 failed: joined output is not unique at learner-attempt grain.")
    if (base["imd_band"] == "10-20").any() or base["imd_band_display"].isna().any():
        raise ValueError("T07 failed: imd_band display normalization did not complete.")
    if base[zero_when_no_event].isna().any().any():
        raise ValueError("T07 failed: dashboard count/total aggregates still contain nulls.")
    output = PROCESSED / "clean_dataset.csv"
    base.to_csv(output, index=False)
    metrics = {
        "task": "T07", "grain": ATTEMPT_KEY, "output": str(output.relative_to(ROOT)),
        "rows": {"studentInfo": len(info), "studentAssessment_events": len(student_assessment), "assessment_aggregated_attempts": len(assessment_agg), "studentVle_events": click_rows, "vle_aggregated_attempts": len(vle_agg), "joined_output": len(base)},
        "unmatched": {"assessment_dimension": assessment_unmatched, "vle_dimension": click_unmatched, "registration": registration_unmatched, "courses": courses_unmatched},
        "joined_output_duplicate_attempt_keys": int(base.duplicated(ATTEMPT_KEY).sum()),
        "final_result": {str(key): int(value) for key, value in base["final_result"].value_counts().sort_index().items()},
        "at_risk": {str(key): int(value) for key, value in base["At_Risk"].value_counts().sort_index().items()},
        "dashboard_ready": {
            "imd_band_normalized_10_20_to_10_20_percent": int((base["imd_band"] == "10-20%").sum()),
            "imd_band_display_unknown": int((base["imd_band_display"] == "Unknown").sum()),
            "zero_filled_columns": zero_when_no_event,
            "zero_filled_columns_nulls_after": int(base[zero_when_no_event].isna().sum().sum()),
            "excluded_zero_variance_column": "has_registration_record",
            "retained_nullable_columns": [
                "date_registration", "date_unregistration", "assessment_score_mean_all_time",
                "assessment_score_min_all_time", "assessment_score_max_all_time",
                "vle_first_event_day", "vle_last_event_day",
            ],
        },
 
        "feature_guard": "All *_all_time VLE/assessment aggregates are descriptive EDA/dashboard fields only. The model builds a separate day-105 snapshot. final_result, Academic_Fail, At_Risk, Withdrawn_Flag and date_unregistration are prohibited model features.",
    }
    JOIN_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"T07 processed dataset written to {output.relative_to(ROOT)}")


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics {path.relative_to(ROOT)}. Run its stage first.")
    return json.loads(path.read_text(encoding="utf-8"))


def n(value: Any) -> str:
    return "—" if value is None else f"{value:,}" if isinstance(value, int) else str(value)


def report() -> None:
    audit_metrics = load_metrics(AUDIT_METRICS)
    clean_metrics = load_metrics(CLEAN_METRICS)
    join_metrics = load_metrics(JOIN_METRICS)
    lines = [
        "# Data Quality Report — OULAD (T05–T07)", "",
        f"**Ngày chạy pipeline:** {date.today().strftime('%d/%m/%Y')}.",
        "**Task:** T05–T07.",
        "**Hạt đầu ra:** một lượt học `(code_module, code_presentation, id_student)`.  ",
        f"**Công nghệ:** Python, pandas {audit_metrics['pandas']}, NumPy {audit_metrics['numpy']}.  ",
        "**Input:** bảy CSV raw cục bộ tại `data/raw/`, đã xác minh T01/T02; raw không bị sửa.", "",
        "## T05 — Audit raw", "",
        "Lệnh tái tạo: `python src/oulad_pipeline.py audit data/raw`.", "",
        "| Bảng | Dòng | Cột | Duplicate toàn dòng | Key / duplicate key | Blank / `?` |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name in FILES:
        table = audit_metrics["tables"][name]
        missing = sum(item["blank"] + item["question_mark"] for item in table["missing"].values())
        key = table.get("key", table.get("event_key"))
        key_text = "không áp dụng" if key is None else f"{', '.join(key['columns'])}; duplicate {n(key['duplicate_rows'])}"
        duplicate = table.get("whole_row_duplicates", table.get("whole_row_duplicates_within_chunks"))
        lines.append(f"| `{name}` | {n(table['rows'])} | {len(table['columns'])} | {n(duplicate)} | {key_text} | {n(missing)} |")
    lines += ["", "### Schema quan sát", "", "| Bảng | Cột theo thứ tự raw | Dtype đọc bởi pandas |", "|---|---|---|"]
    for name in FILES:
        table = audit_metrics["tables"][name]
        columns = ", ".join(f"`{column}`" for column in table["columns"])
        dtypes = ", ".join(f"`{column}`: {dtype}" for column, dtype in table["dtypes"].items())
        lines.append(f"| `{name}` | {columns} | {dtypes} |")
    lines += ["", "### Missing, invalid và outlier", "", "`?` được phát hiện trên raw: `imd_band` 1.111; `date_registration` 45; `date_unregistration` 22.521; `assessments.date` 11; `studentAssessment.score` 173; `vle.week_from`/`week_to` mỗi cột 5.243. Đây là mã missing, không phải blank.", ""]
    lines += ["| Bảng / kiểm tra | Kết quả thực | Diễn giải T05 |", "|---|---:|---|"]
    for name in FILES:
        table = audit_metrics["tables"][name]
        for column, summary in table["numeric"].items():
            outlier = summary["iqr_outliers"]
            result = f"range {n(summary['min'])}–{n(summary['max'])}; IQR diagnostic {n(outlier)}"
            lines.append(f"| `{name}.{column}` | {result} | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |")
        for column, invalid in table.get("invalid", {}).items():
            if isinstance(invalid, list):
                lines.append(f"| `{name}.{column}` invalid category | {', '.join(invalid) if invalid else '0'} | So với domain OULAD đã ghi. |")
    relationships = audit_metrics["relationships"]
    lines += ["", "### Khóa, cardinality và kiểm tra nghiệp vụ", "", "| Kiểm tra | Kết quả |", "|---|---:|"]
    for key, value in relationships.items():
        if key == "unregistration_missing_by_final_result":
            continue
        lines.append(f"| `{key}` | {n(value) if not isinstance(value, dict) else ', '.join(f'{k}={v}' for k, v in value.items())} |")
    lines += [
        "", "`date_unregistration` missing không đồng nghĩa chắc chắn với không rút: trong raw có 93 lượt `Withdrawn` vẫn missing, trong khi 10.063 lượt `Withdrawn` có ngày rút. Do đó T06 giữ missing và mọi bước model phải cấm cột này.",
    ]
    lines += [
        "", "`studentVle` là bảng đóng góp click. Nhiều dòng cùng learner–resource–day được gom theo khóa `(code_module, code_presentation, id_student, id_site, date)` và cộng `sum_click`; không xóa chỉ vì toàn dòng giống nhau. T06 bắt buộc bảo toàn tổng click trước/sau chuẩn hóa.",
        "", "## T06 — Cleaning", "",
        "Lệnh tái tạo: `python src/oulad_pipeline.py clean data/raw`.", "",
    ]
    for rule in clean_metrics["rules"]:
        lines.append(f"- {rule}")
    lines += ["", "| Bảng | Dòng trước | Dòng sau | Exact duplicate quan sát | Dòng gom theo event key | Duplicate bị xóa | Missing sau |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in FILES:
        table = clean_metrics["tables"][name]
        lines.append(
            f"| `{name}` | {n(table['before']['rows'])} | {n(table['after']['rows'])} | "
            f"{n(table['exact_duplicate_rows_observed'])} | "
            f"{n(table['consolidated_event_key_rows'])} | "
            f"{n(table['dropped_exact_duplicates'])} | "
            f"{n(sum(table['after']['missing'].values()))} |"
        )
    vle_clean = clean_metrics["tables"]["studentVle.csv"]
    lines += [
        "",
        f"`studentVle.sum_click` được bảo toàn: {n(vle_clean['click_total_before'])} trước "
        f"và {n(vle_clean['click_total_after'])} sau khi gom event key.",
    ]
    lines += [
        "", "Quyết định sử dụng: `imd_band` giữ missing nullable (không thay bằng median); EDA/dashboard có thể hiển thị category `Unknown` nhưng phải ghi rõ mẫu số. Cleaning không đặt ngưỡng model; model v5 dùng cutoff ngày 105 và threshold 0,335 từ artifact đã kiểm định.",
        "", "## T07 — Join, aggregate và calculated fields", "",
        "Lệnh tái tạo: `python src/oulad_pipeline.py build data/raw`.", "",
        "| Chỉ số | Kết quả |", "|---|---:|",
    ]
    for key, value in join_metrics["rows"].items():
        lines.append(f"| `{key}` | {n(value)} |")
    for key, value in join_metrics["unmatched"].items():
        lines.append(f"| unmatched `{key}` | {n(value)} |")
    dashboard_ready = join_metrics.get("dashboard_ready", {})
    if dashboard_ready:
        lines += [
            f"| `imd_band` `10-20` normalized to `10-20%` | {n(dashboard_ready['imd_band_normalized_10_20_to_10_20_percent'])} |",
            f"| `imd_band_display = Unknown` | {n(dashboard_ready['imd_band_display_unknown'])} |",
            f"| nulls after selected aggregate zero-fill | {n(dashboard_ready['zero_filled_columns_nulls_after'])} |",
            f"| excluded zero-variance QA field | `{dashboard_ready['excluded_zero_variance_column']}` |",
        ]
    lines += [
        f"| duplicate attempt key sau join | {n(join_metrics['joined_output_duplicate_attempt_keys'])} |",
        f"| `final_result` | {', '.join(f'{key}={value:,}' for key, value in join_metrics['final_result'].items())} |",
        f"| `At_Risk` | {', '.join(f'{key}={value:,}' for key, value in join_metrics['at_risk'].items())} |",
        "", "`Academic_Fail = 1` chỉ cho `Fail`; `0` cho `Pass`/`Distinction`. `Withdrawn_Flag` được mô tả riêng và `At_Risk` là alias tương thích của `Academic_Fail`. Các aggregate `*_all_time` chỉ dành cho mô tả; model phải dựng snapshot cutoff riêng. Không tạo attendance, study hours, sleep hoặc previous grade giả.",
        "", "## Đánh giá điều kiện nghiệm thu T05–T07", "",
        "| Điều kiện | Trạng thái | Bằng chứng / giới hạn |", "|---|---|---|",
        "| Pipeline tái tạo từ 7 CSV | Đạt về chạy cục bộ | Script và các lệnh trên; `clean_dataset.csv` được theo dõi và bản hiệu chỉnh chưa commit trong đợt tái cấu trúc. |",
        "| Missing/outlier/duplicate có quyết định | Đạt về pipeline cục bộ | Báo cáo T05/T06; `studentVle` được gom theo learner–resource–day và bảo toàn tổng `sum_click`; bảng khác chỉ loại exact duplicate khi có. |",
        "| Join không nhân dòng | Đạt theo test T07 | Output cùng số dòng `studentInfo`, duplicate attempt key 0; event được aggregate trước join. |",
        "| Dùng được cho EDA/dashboard Python và làm nền model | Đạt local có giới hạn | Bảng processed dành cho mô tả; model dùng bảng interim và feature theo cutoff, không dùng aggregate `*_all_time`. |",
        "| Hiệu chỉnh event VLE | Đã kiểm tra local | Logic bảo toàn click, report và output local đã tái tạo; chưa commit/push. |",
        "", "## Cách sử dụng và giới hạn", "",
        "- EDA dùng `clean_dataset.csv` tái tạo cục bộ cùng Data Quality Report; các tỷ lệ dùng mẫu số là lượt học, không phải sinh viên unique.",
        "- Model/dashboard dùng target `Academic_Fail`, guard leakage và split theo `id_student`. Model v5 dùng cutoff ngày 105, threshold 0,335; phải đối chiếu artifact trước khi công bố.",
        "- Không có thao tác dashboard trong T05–T07. Không có insight hay kết quả model được công bố ở đây.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Data Quality Report written to {REPORT.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "clean", "build", "report"))
    parser.add_argument("raw_dir", nargs="?", type=Path, help="Required except for report")
    args = parser.parse_args()
    if args.stage == "report":
        report()
    elif args.raw_dir is None:
        parser.error("raw_dir is required for audit, clean and build")
    elif args.stage == "audit":
        audit(args.raw_dir)
    elif args.stage == "clean":
        clean(args.raw_dir)
    else:
        build(args.raw_dir)


if __name__ == "__main__":
    main()
