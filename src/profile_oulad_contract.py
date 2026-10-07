"""Profile and validate the local OULAD CSV contract for task T02.

Example:
    python src/profile_oulad_contract.py data/raw

The script only reads the seven source CSV files.  It reports their observed
schema, missingness, candidate value domains, key checks, and join coverage.
It intentionally does not declare ``studentVle`` unique at event grain.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path


FILES = (
    "courses.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "assessments.csv",
    "studentAssessment.csv",
    "vle.csv",
    "studentVle.csv",
)
KEYS = {
    "courses.csv": ("code_module", "code_presentation"),
    "studentInfo.csv": ("code_module", "code_presentation", "id_student"),
    "studentRegistration.csv": ("code_module", "code_presentation", "id_student"),
    "assessments.csv": ("id_assessment",),
    "studentAssessment.csv": ("id_assessment", "id_student"),
    "vle.csv": ("id_site",),
}
NUMBER = re.compile(r"^[+-]?(?:\d+|\d*\.\d+)$")


def value_kind(value: str) -> str:
    if not NUMBER.match(value):
        return "string"
    return "integer" if "." not in value else "number"


def profile(path: Path) -> tuple[list[str], int, dict[str, dict[str, object]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        columns = reader.fieldnames
        stats = {
            column: {"missing": 0, "unknown": 0, "kinds": set(), "values": set(), "overflow": False,
                     "min": None, "max": None}
            for column in columns
        }
        rows = 0
        for row in reader:
            rows += 1
            for column in columns:
                value = row[column]
                stat = stats[column]
                if value == "":
                    stat["missing"] = int(stat["missing"]) + 1
                    continue
                if value == "?":
                    stat["unknown"] = int(stat["unknown"]) + 1
                    continue
                kind = value_kind(value)
                stat["kinds"].add(kind)
                values = stat["values"]
                if len(values) < 30:
                    values.add(value)
                else:
                    stat["overflow"] = True
                if kind != "string":
                    numeric = float(value)
                    stat["min"] = numeric if stat["min"] is None else min(stat["min"], numeric)
                    stat["max"] = numeric if stat["max"] is None else max(stat["max"], numeric)
        return columns, rows, stats


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        yield from csv.DictReader(source)


def tuples(path: Path, fields: tuple[str, ...]):
    return [tuple(row[field] for field in fields) for row in rows(path)]


def key_result(name: str, values: list[tuple[str, ...]]) -> tuple[int, int, int]:
    nulls = sum(any(value == "" for value in item) for item in values)
    unique = len(set(values))
    return nulls, unique, len(values) - unique


def fmt_number(value: float | None) -> str:
    if value is None:
        return "—"
    return str(int(value)) if value.is_integer() else f"{value:g}"


def display_domain(stat: dict[str, object]) -> str:
    values = sorted(str(value) for value in stat["values"])
    if stat["overflow"]:
        return "Nhiều giá trị; xem T05"
    return ", ".join(f"`{value}`" for value in values) or "—"


def main() -> int:
    # PowerShell on Windows may otherwise use cp1252 for redirected output.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    args = parser.parse_args()
    root = args.raw_dir
    missing = [name for name in FILES if not (root / name).is_file()]
    if missing:
        print(f"Missing CSV: {', '.join(missing)}", file=sys.stderr)
        return 1

    observed = {}
    for name in FILES:
        observed[name] = profile(root / name)

    print("# T02 OULAD data-contract verification")
    print()
    print("## Observed schemas")
    for name in FILES:
        columns, count, stats = observed[name]
        print(f"\n### {name} ({count:,} rows; {len(columns)} columns)")
        print("| Column | Observed kind | Blank / `?` | Observed range / domain |")
        print("|---|---|---:|---|")
        for column in columns:
            stat = stats[column]
            kinds = "/".join(sorted(stat["kinds"])) or "empty"
            if stat["kinds"] and stat["kinds"] <= {"integer", "number"}:
                domain = f"{fmt_number(stat['min'])} to {fmt_number(stat['max'])}"
            else:
                domain = display_domain(stat)
            missing = f"{int(stat['missing']):,} / {int(stat['unknown']):,}"
            print(f"| `{column}` | {kinds} | {missing} | {domain} |")

    course_keys = set(tuples(root / "courses.csv", KEYS["courses.csv"]))
    info_keys = set(tuples(root / "studentInfo.csv", KEYS["studentInfo.csv"]))
    registration_keys = set(tuples(root / "studentRegistration.csv", KEYS["studentRegistration.csv"]))
    assessment_rows = list(rows(root / "assessments.csv"))
    assessment_by_id = {row["id_assessment"]: row for row in assessment_rows}
    vle_rows = list(rows(root / "vle.csv"))
    vle_by_key = {(row["code_module"], row["code_presentation"], row["id_site"]): row for row in vle_rows}

    print("\n## Candidate-key checks")
    print("| Table | Candidate key | Null key rows | Distinct keys | Duplicate rows | Result |")
    print("|---|---|---:|---:|---:|---|")
    for name, fields in KEYS.items():
        nulls, unique, duplicates = key_result(name, tuples(root / name, fields))
        result = "PASS" if nulls == duplicates == 0 else "FAIL"
        print(f"| `{name}` | {', '.join(f'`{field}`' for field in fields)} | {nulls:,} | {unique:,} | {duplicates:,} | {result} |")
    print("| `studentVle.csv` | `(code_module, code_presentation, id_student, id_site, date)` | checked below | not asserted unique | not asserted unique | N/A: event rows may repeat |")

    registration_outside_info = len(registration_keys - info_keys)
    info_without_registration = len(info_keys - registration_keys)
    assessment_outside_courses = sum(
        (row["code_module"], row["code_presentation"]) not in course_keys for row in assessment_rows
    )
    vle_outside_courses = sum(
        (row["code_module"], row["code_presentation"]) not in course_keys for row in vle_rows
    )
    assessment_unmatched = 0
    assessment_student_unmatched = 0
    for row in rows(root / "studentAssessment.csv"):
        assessment = assessment_by_id.get(row["id_assessment"])
        if assessment is None:
            assessment_unmatched += 1
        elif (assessment["code_module"], assessment["code_presentation"], row["id_student"]) not in info_keys:
            assessment_student_unmatched += 1
    vle_site_unmatched = 0
    vle_student_unmatched = 0
    vle_key_nulls = 0
    for row in rows(root / "studentVle.csv"):
        event_key = (row["code_module"], row["code_presentation"], row["id_student"], row["id_site"], row["date"])
        vle_key_nulls += any(value == "" for value in event_key)
        if (row["code_module"], row["code_presentation"], row["id_site"]) not in vle_by_key:
            vle_site_unmatched += 1
        if (row["code_module"], row["code_presentation"], row["id_student"]) not in info_keys:
            vle_student_unmatched += 1

    print("\n## Join coverage checks")
    print("| Relationship / check | Unmatched or invalid rows | Result |")
    print("|---|---:|---|")
    for label, count in (
        ("studentRegistration learner-attempt -> studentInfo", registration_outside_info),
        ("studentInfo learner-attempt without studentRegistration", info_without_registration),
        ("assessments module-presentation -> courses", assessment_outside_courses),
        ("vle module-presentation -> courses", vle_outside_courses),
        ("studentAssessment id_assessment -> assessments", assessment_unmatched),
        ("studentAssessment inferred learner-attempt -> studentInfo", assessment_student_unmatched),
        ("studentVle site/module-presentation -> vle", vle_site_unmatched),
        ("studentVle learner-attempt -> studentInfo", vle_student_unmatched),
        ("studentVle null event-key components", vle_key_nulls),
    ):
        print(f"| {label} | {count:,} | {'PASS' if count == 0 else 'CHECK'} |")

    final_result = Counter(row["final_result"] for row in rows(root / "studentInfo.csv"))
    regions = {row["region"] for row in rows(root / "studentInfo.csv") if row["region"]}
    print("\n## T02 target and map fields")
    print("- `final_result`: " + ", ".join(f"`{key}`={value:,}" for key, value in sorted(final_result.items())))
    print(f"- Non-empty `region` values: {len(regions)}")
    print("- `Academic_Fail` is derived: Fail = 1, Pass/Distinction = 0; Withdrawn is analysed separately and excluded from the academic model.")
    print("- No cleaning, aggregation, calculated field, model feature, or dashboard comparison is performed by T02.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
