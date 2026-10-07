"""Screen prediction checkpoints without looking at the test split.

This script uses one fixed Logistic Regression configuration for every cutoff.
It exists only to choose the earliest viable checkpoint from validation data;
hyperparameter search, bootstrap confidence intervals and final test reporting
remain the responsibility of ``at_risk_model.py train``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from at_risk_features import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    build_feature_snapshot,
)
from at_risk_model import (
    build_estimator,
    choose_threshold,
    create_group_splits,
    evaluate_predictions,
)


ROOT = Path(__file__).resolve().parents[1]


def screen_cutoff(input_dir: Path, cutoff: int, random_state: int) -> dict[str, float | int]:
    snapshot, counts = build_feature_snapshot(input_dir, cutoff)
    snapshot["dataset_split"] = create_group_splits(snapshot, random_state)
    features = MODEL_CATEGORICAL_FEATURES + MODEL_NUMERIC_FEATURES
    train = snapshot["dataset_split"].eq("train")
    validation = snapshot["dataset_split"].eq("validation")

    estimator = build_estimator().set_params(
        model__C=0.3,
        model__class_weight=None,
        model__l1_ratio=1.0,
    )
    estimator.fit(snapshot.loc[train, features], snapshot.loc[train, "At_Risk"])
    probabilities = estimator.predict_proba(snapshot.loc[validation, features])[:, 1]
    threshold, _ = choose_threshold(
        snapshot.loc[validation, "At_Risk"].to_numpy(),
        probabilities,
        minimum_recall=0.70,
    )
    metrics = evaluate_predictions(
        snapshot.loc[validation, "At_Risk"].to_numpy(), probabilities, threshold
    )
    return {
        "cutoff_day": cutoff,
        "eligible_attempts": counts["eligible_attempts"],
        "excluded_withdrawn_outcome": counts["excluded_withdrawn_outcome"],
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only screening for OULAD Academic-Fail cutoffs."
    )
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "interim"))
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[30, 60, 90, 105])
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "processed" / "model_academic_fail" / "cutoff_screening.csv"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    rows = [
        screen_cutoff(Path(args.input_dir), cutoff, args.random_state)
        for cutoff in sorted(set(args.cutoffs))
    ]
    result = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    display_columns = [
        "cutoff_day",
        "rows",
        "at_risk_rate",
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "precision_at_risk",
        "recall_at_risk",
        "f1_at_risk",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]
    print(result[display_columns].to_string(index=False))
    print(f"Validation-only screening written to {output.resolve()}")


if __name__ == "__main__":
    main()
