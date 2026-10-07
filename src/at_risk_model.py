"""Train and evaluate the OULAD early-warning Logistic Regression model.

This command-line module produces local, reproducible artifacts for model QA
and later Streamlit dashboard integration.  It never trains on the all-time aggregates in
``clean_dataset.csv``.

Examples (PowerShell):
    python src/at_risk_model.py audit-cutoffs
    python src/at_risk_model.py train --cutoff 105
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from at_risk_features import (
    ATTEMPT_KEY,
    AUDIT_COLUMNS,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_INTERACTION_FEATURES,
    MODEL_NUMERIC_FEATURES,
    PROHIBITED_MODEL_FEATURES,
    audit_cutoffs,
    build_feature_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "interim"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "model"
DEFAULT_MODEL = ROOT / "models" / "logistic_regression.joblib"
DEFAULT_REPORT = DEFAULT_OUTPUT / "model_evaluation.txt"
MODEL_NAME = "logistic_regression"
BASELINE_NAME = "dummy_prior_baseline"
CONFIDENCE_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision_at_risk",
    "recall_at_risk",
    "f1_at_risk",
    "roc_auc",
    "pr_auc",
    "brier_score",
)
VERIFICATION_THRESHOLDS = {
    "minimum_accuracy": 0.80,
    "minimum_accuracy_ci_lower": 0.80,
    "minimum_recall": 0.70,
    "minimum_balanced_accuracy": 0.78,
    "minimum_f1": 0.70,
    "minimum_roc_auc": 0.85,
    "minimum_pr_auc_margin": 0.20,
    "maximum_brier": 0.15,
    "minimum_baseline_margin": 0.15,
    "minimum_presentation_accuracy": 0.80,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_group_splits(
    snapshot: pd.DataFrame,
    random_state: int,
) -> pd.Series:
    """Create approximately 71/14/14 splits with disjoint student groups."""
    y = snapshot["At_Risk"].to_numpy()
    groups = snapshot["id_student"].to_numpy()
    placeholder = np.zeros((len(snapshot), 1), dtype=np.int8)

    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=random_state)
    train_validation_idx, test_idx = next(outer.split(placeholder, y, groups))

    inner_y = y[train_validation_idx]
    inner_groups = groups[train_validation_idx]
    inner_placeholder = np.zeros((len(train_validation_idx), 1), dtype=np.int8)
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=random_state + 1)
    train_rel_idx, validation_rel_idx = next(
        inner.split(inner_placeholder, inner_y, inner_groups)
    )
    train_idx = train_validation_idx[train_rel_idx]
    validation_idx = train_validation_idx[validation_rel_idx]

    split = pd.Series(index=snapshot.index, dtype="string")
    split.loc[snapshot.index[train_idx]] = "train"
    split.loc[snapshot.index[validation_idx]] = "validation"
    split.loc[snapshot.index[test_idx]] = "test"
    if split.isna().any():
        raise AssertionError("Some rows were not assigned to a dataset split.")

    group_sets = {
        name: set(snapshot.loc[split.eq(name), "id_student"].astype(int))
        for name in ("train", "validation", "test")
    }
    if (
        group_sets["train"] & group_sets["validation"]
        or group_sets["train"] & group_sets["test"]
        or group_sets["validation"] & group_sets["test"]
    ):
        raise AssertionError("id_student leakage detected across dataset splits.")
    return split


def build_estimator() -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pairwise_interactions = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "interactions",
                PolynomialFeatures(
                    degree=(2, 2), interaction_only=True, include_bias=False
                ),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric, MODEL_NUMERIC_FEATURES),
            ("categorical", categorical, MODEL_CATEGORICAL_FEATURES),
            ("pairwise_interactions", pairwise_interactions, MODEL_INTERACTION_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=2_000,
                    random_state=42,
                ),
            ),
        ]
    )


def choose_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    minimum_recall: float,
) -> tuple[float, pd.DataFrame]:
    """Maximize validation accuracy subject to the At-Risk recall floor."""
    rows: list[dict[str, float | bool]] = []
    # A 0.001 grid avoids leaving measurable validation accuracy on the table
    # while remaining deterministic and much less prone to chasing individual
    # probability values than testing every unique prediction as a threshold.
    for threshold in np.linspace(0.05, 0.95, 901):
        predicted = probabilities >= threshold
        recall = float(recall_score(y_true, predicted, zero_division=0))
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
                "precision": float(precision_score(y_true, predicted, zero_division=0)),
                "recall": recall,
                "f1": float(f1_score(y_true, predicted, zero_division=0)),
                "meets_minimum_recall": bool(recall >= minimum_recall),
            }
        )
    table = pd.DataFrame(rows)
    eligible = table.loc[table["meets_minimum_recall"]].copy()
    if eligible.empty:
        eligible = table.copy()
    selected = eligible.sort_values(
        ["accuracy", "f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, False, False],
    ).iloc[0]
    table["selected"] = np.isclose(table["threshold"], selected["threshold"])
    return float(selected["threshold"]), table


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )
    brier = float(brier_score_loss(y_true, probabilities))
    model_log_loss = float(log_loss(y_true, probabilities, labels=[0, 1]))
    null_probability = np.full(len(y_true), float(np.mean(y_true)))
    null_log_loss = float(log_loss(y_true, null_probability, labels=[0, 1]))
    return {
        "rows": int(len(y_true)),
        "at_risk_rate": float(np.mean(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision_at_risk": float(precision_score(y_true, predicted, zero_division=0)),
        "recall_at_risk": float(recall_score(y_true, predicted, zero_division=0)),
        "specificity": float(specificity),
        "f1_at_risk": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": brier,
        "probability_rmse": float(np.sqrt(brier)),
        "probability_mae": float(np.mean(np.abs(y_true - probabilities))),
        "log_loss": model_log_loss,
        "null_log_loss": null_log_loss,
        "mcfadden_pseudo_r2": float(1.0 - model_log_loss / null_log_loss),
        "matthews_correlation": float(matthews_corrcoef(y_true, predicted)),
        "cohen_kappa": float(cohen_kappa_score(y_true, predicted)),
        "false_negatives": int(false_negative),
        "false_positives": int(false_positive),
    }


def make_risk_band(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Create explicit bands tied to the validated classification threshold."""
    medium_floor = threshold / 2.0
    return np.select(
        [probabilities >= threshold, probabilities >= medium_floor],
        ["High", "Medium"],
        default="Low",
    )


def make_error_type(y_true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.select(
        [
            (y_true == 1) & (predicted == 1),
            (y_true == 0) & (predicted == 0),
            (y_true == 0) & (predicted == 1),
            (y_true == 1) & (predicted == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="UNKNOWN",
    )


def _split_summary(snapshot: pd.DataFrame) -> pd.DataFrame:
    return (
        snapshot.groupby("dataset_split", observed=True)
        .agg(
            rows=("At_Risk", "size"),
            at_risk_count=("At_Risk", "sum"),
            at_risk_rate=("At_Risk", "mean"),
            distinct_students=("id_student", "nunique"),
        )
        .reset_index()
    )


def _coefficient_table(estimator: Pipeline, model_version: str) -> pd.DataFrame:
    preprocessing = estimator.named_steps["preprocessing"]
    model = estimator.named_steps["model"]
    names = preprocessing.get_feature_names_out()
    coefficients = model.coef_[0]
    table = pd.DataFrame(
        {
            "feature": [name.replace("numeric__", "").replace("categorical__", "") for name in names],
            "coefficient": coefficients,
        }
    )
    table["odds_ratio"] = np.exp(table["coefficient"].clip(-20, 20))
    table["absolute_coefficient"] = table["coefficient"].abs()
    table["direction"] = np.where(table["coefficient"].ge(0), "higher_risk", "lower_risk")
    table["model_version"] = model_version
    return table.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)


def _confusion_rows(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    cutoff_day: int,
    model_version: str,
) -> list[dict[str, object]]:
    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    rows: list[dict[str, object]] = []
    for actual in (0, 1):
        for prediction in (0, 1):
            rows.append(
                {
                    "model_name": model_name,
                    "dataset_split": split_name,
                    "actual_at_risk": actual,
                    "predicted_at_risk": prediction,
                    "count": int(matrix[actual, prediction]),
                    "threshold": threshold,
                    "cutoff_day": cutoff_day,
                    "model_version": model_version,
                }
            )
    return rows


def _subgroup_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute test-only QA metrics for dashboard filters and fairness checks."""
    test = predictions.loc[predictions["dataset_split"].eq("test")].copy()
    rows: list[dict[str, object]] = []
    dimensions = [
        "code_module",
        "code_presentation",
        "region",
        "gender",
        "age_band",
        "disability",
        "imd_band",
    ]
    for dimension in dimensions:
        values = test[dimension].fillna("Unknown")
        for value, index in values.groupby(values, observed=True).groups.items():
            group = test.loc[index]
            actual = group["actual_at_risk"].to_numpy()
            predicted = group["predicted_at_risk"].to_numpy()
            probability = group["risk_probability"].to_numpy()
            has_both_classes = np.unique(actual).size == 2
            rows.append(
                {
                    "dimension": dimension,
                    "group_value": str(value),
                    "rows": int(len(group)),
                    "actual_at_risk_rate": float(actual.mean()),
                    "predicted_at_risk_rate": float(predicted.mean()),
                    "accuracy": float(accuracy_score(actual, predicted)),
                    "precision_at_risk": float(
                        precision_score(actual, predicted, zero_division=0)
                    ),
                    "recall_at_risk": float(recall_score(actual, predicted, zero_division=0)),
                    "f1_at_risk": float(f1_score(actual, predicted, zero_division=0)),
                    "roc_auc": float(roc_auc_score(actual, probability))
                    if has_both_classes
                    else np.nan,
                    "pr_auc": float(average_precision_score(actual, probability))
                    if has_both_classes
                    else np.nan,
                    "false_negatives": int(((actual == 1) & (predicted == 0)).sum()),
                }
            )
    return pd.DataFrame(rows)


def _curve_points(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    model_version: str,
) -> pd.DataFrame:
    false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(
        y_true, probabilities
    )
    precision, recall, pr_thresholds = precision_recall_curve(y_true, probabilities)
    roc_table = pd.DataFrame(
        {
            "curve": "ROC",
            "x": false_positive_rate,
            "y": true_positive_rate,
            "threshold": roc_thresholds,
        }
    )
    pr_table = pd.DataFrame(
        {
            "curve": "Precision-Recall",
            "x": recall,
            "y": precision,
            "threshold": np.append(pr_thresholds, np.nan),
        }
    )
    table = pd.concat([roc_table, pr_table], ignore_index=True)
    table["dataset_split"] = "test"
    table["model_name"] = MODEL_NAME
    table["model_version"] = model_version
    return table


def _calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    test = predictions.loc[predictions["dataset_split"].eq("test")].copy()
    test["probability_bin"] = pd.qcut(
        test["risk_probability"], q=10, duplicates="drop"
    )
    calibration = (
        test.groupby("probability_bin", observed=True)
        .agg(
            rows=("actual_at_risk", "size"),
            mean_predicted_probability=("risk_probability", "mean"),
            observed_at_risk_rate=("actual_at_risk", "mean"),
        )
        .reset_index()
    )
    calibration["probability_bin"] = calibration["probability_bin"].astype(str)
    calibration["dataset_split"] = "test"
    return calibration


def _group_bootstrap_confidence_intervals(
    predictions: pd.DataFrame,
    threshold: float,
    model_version: str,
    iterations: int = 2_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate test-metric uncertainty by resampling whole students.

    Some learners have more than one attempt. Resampling rows independently
    would treat those correlated attempts as unrelated, so the bootstrap unit
    is ``id_student`` and every sampled learner contributes all of their rows.
    """
    test = predictions.loc[predictions["dataset_split"].eq("test")].reset_index(
        drop=True
    )
    group_indices = [
        values.to_numpy(dtype=int)
        for values in test.groupby("id_student", sort=False).groups.values()
    ]
    if not group_indices:
        raise ValueError("Cannot bootstrap an empty test set.")

    actual = test["actual_at_risk"].to_numpy(dtype=int)
    probabilities = test["risk_probability"].to_numpy(dtype=float)
    point = evaluate_predictions(actual, probabilities, threshold)
    rng = np.random.default_rng(random_state)
    samples = {metric: np.empty(iterations, dtype=float) for metric in CONFIDENCE_METRICS}

    for iteration in range(iterations):
        selected_groups = rng.integers(0, len(group_indices), size=len(group_indices))
        selected_rows = np.concatenate([group_indices[index] for index in selected_groups])
        result = evaluate_predictions(
            actual[selected_rows], probabilities[selected_rows], threshold
        )
        for metric in CONFIDENCE_METRICS:
            samples[metric][iteration] = float(result[metric])

    rows = []
    for metric in CONFIDENCE_METRICS:
        lower, upper = np.quantile(samples[metric], [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "point_estimate": float(point[metric]),
                "lower_95": float(lower),
                "upper_95": float(upper),
                "bootstrap_unit": "id_student",
                "bootstrap_iterations": iterations,
                "dataset_split": "test",
                "model_name": MODEL_NAME,
                "model_version": model_version,
            }
        )
    return pd.DataFrame(rows)


def write_model_report(
    path: Path,
    cutoff_day: int,
    cohort_counts: dict[str, int],
    split_summary: pd.DataFrame,
    best_params: dict[str, Any],
    threshold: float,
    minimum_recall: float,
    metrics: pd.DataFrame,
    confidence_intervals: pd.DataFrame,
    coefficients: pd.DataFrame,
    output_hashes: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logistic_test = metrics.loc[
        metrics["model_name"].eq(MODEL_NAME) & metrics["dataset_split"].eq("test")
    ].iloc[0]
    baseline_test = metrics.loc[
        metrics["model_name"].eq(BASELINE_NAME) & metrics["dataset_split"].eq("test")
    ].iloc[0]
    lines = [
        "# Model Evaluation — OULAD Academic-Fail Logistic Regression",
        "",
        f"**Ngày chạy (UTC):** {datetime.now(timezone.utc).isoformat(timespec='seconds')}.  ",
        f"**Mốc dự báo:** ngày {cutoff_day} tính từ đầu presentation.  ",
        "**Target:** `Academic_Fail = 1` cho `Fail`; `0` cho `Pass/Distinction`; "
        "`Withdrawn` được loại khỏi model học thuật và phân tích riêng.  ",
        "**Đơn vị:** một lượt học `(code_module, code_presentation, id_student)`.  ",
        "**Thuật toán chính:** Logistic Regression; `DummyClassifier` chỉ là baseline.  ",
        "",
        "## Cohort tại cutoff",
        "",
        "| Chỉ số | Giá trị |",
        "|---|---:|",
    ]
    for key, value in cohort_counts.items():
        lines.append(f"| `{key}` | {value:,} |")
    lines += [
        "",
        "Các lượt đăng ký sau cutoff, đã rút trước/tại cutoff hoặc có kết quả cuối "
        "`Withdrawn` bị loại khỏi cohort model học thuật. `date_unregistration` chỉ dùng "
        "cho eligibility, không đi vào feature.",
        "",
        "## Split không trùng sinh viên",
        "",
        "| Split | Dòng | Sinh viên | Fail | Fail rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in split_summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset_split} | {row.rows:,} | {row.distinct_students:,} | "
            f"{row.at_risk_count:,} | {row.at_risk_rate:.4f} |"
        )
    lines += [
        "",
        "## Cấu hình đã chọn",
        "",
        f"- `C`: `{best_params['model__C']}`.",
        f"- `class_weight`: `{best_params['model__class_weight']}`.",
        f"- Regularization: `{'L1' if best_params['model__l1_ratio'] == 1.0 else 'L2'}`; "
        "solver: liblinear; preprocessing được fit trong pipeline.",
        f"- Threshold `{threshold:.3f}` tối đa hóa Accuracy trên validation, với "
        f"recall floor `{minimum_recall:.2f}`.",
        "- Feature, hyperparameter và threshold của mỗi lần chạy được chọn bằng train-CV/validation; "
        "lịch sử so sánh cutoff đã xem test và được công bố tại phần giới hạn.",
        "",
        "## Kết quả",
        "",
        "| Model | Split | Accuracy | Balanced Acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.model_name} | {row.dataset_split} | {row.accuracy:.4f} | "
            f"{row.balanced_accuracy:.4f} | {row.precision_at_risk:.4f} | "
            f"{row.recall_at_risk:.4f} | {row.f1_at_risk:.4f} | {row.roc_auc:.4f} | "
            f"{row.pr_auc:.4f} | {row.brier_score:.4f} | {row.false_negatives:,} |"
        )
    lines += [
        "",
        f"Trên test, Logistic Regression có F1 `{logistic_test['f1_at_risk']:.4f}` "
        f"so với baseline `{baseline_test['f1_at_risk']:.4f}`; PR-AUC tương ứng "
        f"`{logistic_test['pr_auc']:.4f}` và `{baseline_test['pr_auc']:.4f}`.",
        "",
        "### Cách đọc bộ metric",
        "",
        "Không metric đơn lẻ nào chứng minh model đáng tin cậy. Accuracy đo tỷ lệ đúng chung; "
        "Balanced Accuracy cân bằng hai lớp; Precision đo mức cảnh báo nhầm; Recall đo mức "
        "bỏ sót Fail; F1 cân bằng Precision/Recall. ROC-AUC đo khả năng xếp hạng qua mọi "
        "threshold, còn PR-AUC tập trung vào lớp Fail và phải so với baseline tỷ lệ lớp. "
        "Brier/Log Loss đánh giá chất lượng xác suất, không chỉ nhãn đúng/sai. MCC/Kappa kiểm "
        "tra mức đồng thuận khi xét toàn bộ confusion matrix.",
        "",
        "Độ tin cậy của kết quả còn dựa vào split không trùng sinh viên, preprocessing chỉ fit "
        "trên train, threshold chỉ chọn trên validation, leakage guard, baseline, khoảng tin cậy "
        "và khả năng tái tạo. Các kiểm tra này cung cấp bằng chứng tổng quát hóa; không tạo ra "
        "bảo đảm tuyệt đối cho dữ liệu tương lai.",
        "",
        "## Chất lượng xác suất và độ đồng thuận trên test",
        "",
        "| Brier | Probability RMSE | Probability MAE | Log Loss | McFadden pseudo-R² | MCC | Cohen Kappa |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {logistic_test['brier_score']:.4f} | {logistic_test['probability_rmse']:.4f} | "
        f"{logistic_test['probability_mae']:.4f} | {logistic_test['log_loss']:.4f} | "
        f"{logistic_test['mcfadden_pseudo_r2']:.4f} | "
        f"{logistic_test['matthews_correlation']:.4f} | {logistic_test['cohen_kappa']:.4f} |",
        "",
        "`R²` thông thường và RMSE của biến liên tục không áp dụng cho target nhị phân. "
        "Probability RMSE là căn Brier Score; pseudo-R² phải luôn ghi đúng tên McFadden.",
        "",
        "## Khoảng tin cậy 95% trên test",
        "",
        "Khoảng dưới đây dùng 2.000 lần bootstrap theo `id_student`, nghĩa là mọi lượt học "
        "của cùng một sinh viên được lấy mẫu cùng nhau. Khoảng này phản ánh độ dao động do "
        "lấy mẫu trong test, không bảo đảm model sẽ giữ nguyên chất lượng khi dữ liệu tương lai "
        "thay đổi phân bố.",
        "",
        "| Metric | Ước lượng | Cận dưới 95% | Cận trên 95% |",
        "|---|---:|---:|---:|",
    ]
    for row in confidence_intervals.itertuples(index=False):
        lines.append(
            f"| `{row.metric}` | {row.point_estimate:.4f} | "
            f"{row.lower_95:.4f} | {row.upper_95:.4f} |"
        )
    lines += [
        "",
        "## Leakage guard",
        "",
        "- Không dùng `id_student`, `final_result`, `Academic_Fail`, `At_Risk`, `Performance_Level` hoặc "
        "`date_unregistration` làm feature.",
        "- Không dùng aggregate `*_all_time`; assessment/VLE đều bị giới hạn tại cutoff.",
        "- Preprocessing và model được fit trên train; threshold chỉ chọn trên validation.",
        "- `gender`, `region`, `imd_band`, `age_band`, `disability` chỉ giữ để QA theo nhóm, "
        "không dùng trong mô hình chính.",
        "",
        "## Hệ số có độ lớn cao nhất",
        "",
        "Hệ số phản ánh liên hệ trong mô hình sau preprocessing, không chứng minh nhân quả.",
        "",
        "| Feature | Coefficient | Odds ratio | Direction |",
        "|---|---:|---:|---|",
    ]
    for row in coefficients.head(15).itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | {row.coefficient:.4f} | {row.odds_ratio:.4f} | {row.direction} |"
        )
    lines += [
        "",
        "## Output cho dashboard Python",
        "",
        "`model_predictions.csv` có một dòng cho mỗi lượt học eligible, gồm actual/predicted, "
        "xác suất, risk band, split, error type, cutoff, threshold và model version. "
        "Dashboard đánh giá phải mặc định `dataset_split = test`.",
        "",
        "Risk band dùng quy tắc minh bạch: `High` nếu xác suất ≥ threshold; `Medium` nếu "
        "xác suất từ nửa threshold đến dưới threshold; còn lại là `Low`.",
        "",
        "| File | SHA-256 |",
        "|---|---|",
    ]
    for name, digest in output_hashes.items():
        lines.append(f"| `{name}` | `{digest}` |")
    lines += [
        "",
        "## Giới hạn",
        "",
        "- Đây là dự báo trên dữ liệu quan sát lịch sử, không phải quan hệ nhân quả hoặc chẩn đoán cá nhân.",
        "- Kết quả phụ thuộc cutoff và coverage assessment khác nhau giữa module/presentation.",
        "- OULAD không có timestamp công bố điểm. Pipeline giả định điểm của bài nộp trước/tại "
        "cutoff đã quan sát được tại cutoff; nếu nghiệp vụ công bố điểm trễ thì phải loại các "
        "feature điểm hoặc dùng timestamp công bố thật rồi train lại.",
        "- CSV dự báo hiện là output đánh giá lịch sử nên chứa actual label; khi dự báo lượt học "
        "mới, `final_result`/target không được truyền vào model.",
        "- Model chỉ hỗ trợ ưu tiên hỗ trợ học tập; không dùng để xử phạt hoặc quyết định tự động.",
        "- Test split đã được xem ở các vòng cải tiến cutoff 98/105; để có đánh giá hoàn toàn "
        "nguyên sơ cần thêm holdout theo thời gian/presentation hoặc dữ liệu ngoài OULAD hiện tại.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cutoff_audit(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = audit_cutoffs(Path(args.input_dir), args.cutoffs, chunksize=args.chunksize)
    audit.to_csv(output, index=False)
    overall = audit.loc[audit["scope"].eq("overall")]
    print(overall.to_string(index=False))
    print(f"Cutoff audit written to {output}")


def run_training(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report).resolve()

    snapshot, cohort_counts = build_feature_snapshot(
        Path(args.input_dir), args.cutoff, chunksize=args.chunksize
    )
    snapshot["dataset_split"] = create_group_splits(snapshot, args.random_state)
    feature_columns = MODEL_CATEGORICAL_FEATURES + MODEL_NUMERIC_FEATURES
    if PROHIBITED_MODEL_FEATURES.intersection(feature_columns):
        raise AssertionError("Prohibited feature in training matrix.")

    train_mask = snapshot["dataset_split"].eq("train")
    validation_mask = snapshot["dataset_split"].eq("validation")
    test_mask = snapshot["dataset_split"].eq("test")
    X = snapshot[feature_columns]
    y = snapshot["At_Risk"].astype(int)

    precision_scorer = make_scorer(precision_score, zero_division=0)
    recall_scorer = make_scorer(recall_score, zero_division=0)
    f1_scorer = make_scorer(f1_score, zero_division=0)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.random_state + 2)
    search = GridSearchCV(
        estimator=build_estimator(),
        param_grid={
            "model__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            "model__class_weight": [None, "balanced"],
            "model__l1_ratio": [0.0, 1.0],
        },
        scoring={
            "pr_auc": "average_precision",
            "roc_auc": "roc_auc",
            "precision": precision_scorer,
            "recall": recall_scorer,
            "f1": f1_scorer,
        },
        refit="pr_auc",
        cv=cv,
        n_jobs=args.n_jobs,
        return_train_score=False,
    )
    search.fit(
        X.loc[train_mask],
        y.loc[train_mask],
        groups=snapshot.loc[train_mask, "id_student"],
    )
    estimator = search.best_estimator_
    validation_probabilities = estimator.predict_proba(X.loc[validation_mask])[:, 1]
    threshold, threshold_table = choose_threshold(
        y.loc[validation_mask].to_numpy(),
        validation_probabilities,
        args.minimum_recall,
    )

    split_probabilities: dict[str, np.ndarray] = {}
    metrics_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    model_version = f"lr-oulad-academic-fail-c{args.cutoff}-s{args.random_state}-v5"

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((int(train_mask.sum()), 1)), y.loc[train_mask])
    for split_name, mask in (
        ("train", train_mask),
        ("validation", validation_mask),
        ("test", test_mask),
    ):
        split_y = y.loc[mask].to_numpy()
        probabilities = estimator.predict_proba(X.loc[mask])[:, 1]
        split_probabilities[split_name] = probabilities
        logistic_metrics = evaluate_predictions(split_y, probabilities, threshold)
        metrics_rows.append(
            {
                "model_name": MODEL_NAME,
                "dataset_split": split_name,
                "cutoff_day": args.cutoff,
                "model_version": model_version,
                **logistic_metrics,
            }
        )
        confusion_rows.extend(
            _confusion_rows(
                MODEL_NAME,
                split_name,
                split_y,
                probabilities,
                threshold,
                args.cutoff,
                model_version,
            )
        )

        baseline_probabilities = dummy.predict_proba(np.zeros((int(mask.sum()), 1)))[:, 1]
        baseline_metrics = evaluate_predictions(split_y, baseline_probabilities, 0.5)
        metrics_rows.append(
            {
                "model_name": BASELINE_NAME,
                "dataset_split": split_name,
                "cutoff_day": args.cutoff,
                "model_version": "dummy-prior-v1",
                **baseline_metrics,
            }
        )
        confusion_rows.extend(
            _confusion_rows(
                BASELINE_NAME,
                split_name,
                split_y,
                baseline_probabilities,
                0.5,
                args.cutoff,
                "dummy-prior-v1",
            )
        )

    probabilities = np.empty(len(snapshot), dtype=float)
    for split_name, mask in (
        ("train", train_mask),
        ("validation", validation_mask),
        ("test", test_mask),
    ):
        probabilities[np.flatnonzero(mask.to_numpy())] = split_probabilities[split_name]
    predicted = (probabilities >= threshold).astype(int)

    prediction_columns = ATTEMPT_KEY + AUDIT_COLUMNS + [
        "final_result",
        "Academic_Fail",
        "At_Risk",
        "actual_status",
        "dataset_split",
    ]
    predictions = snapshot[prediction_columns].copy()
    predictions = predictions.rename(columns={"At_Risk": "actual_at_risk"})
    predictions["risk_probability"] = probabilities
    predictions["predicted_at_risk"] = predicted
    predictions["predicted_status"] = np.where(predicted == 1, "Fail", "Pass/Distinction")
    predictions["actual_fail"] = predictions["actual_at_risk"]
    predictions["failure_probability"] = predictions["risk_probability"]
    predictions["predicted_fail"] = predictions["predicted_at_risk"]
    predictions["risk_band"] = make_risk_band(probabilities, threshold)
    predictions["correct_prediction"] = predictions["actual_at_risk"].to_numpy() == predicted
    predictions["error_type"] = make_error_type(
        predictions["actual_at_risk"].to_numpy(), predicted
    )
    predictions["cutoff_day"] = args.cutoff
    predictions["prediction_threshold"] = threshold
    predictions["model_name"] = MODEL_NAME
    predictions["model_version"] = model_version
    if predictions[ATTEMPT_KEY].duplicated().any():
        raise AssertionError("Prediction output is not unique at learner-attempt grain.")

    metrics = pd.DataFrame(metrics_rows)
    confusion = pd.DataFrame(confusion_rows)
    coefficients = _coefficient_table(estimator, model_version)
    split_summary = _split_summary(snapshot)
    cv_results = pd.DataFrame(search.cv_results_)
    subgroup_metrics = _subgroup_metrics(predictions)
    test_actual = predictions.loc[
        predictions["dataset_split"].eq("test"), "actual_at_risk"
    ].to_numpy()
    test_probabilities = predictions.loc[
        predictions["dataset_split"].eq("test"), "risk_probability"
    ].to_numpy()
    curve_points = _curve_points(test_actual, test_probabilities, model_version)
    calibration = _calibration_table(predictions)
    confidence_intervals = _group_bootstrap_confidence_intervals(
        predictions,
        threshold,
        model_version,
        iterations=2_000,
        random_state=args.random_state + 100,
    )

    paths = {
        "feature_snapshot.csv": output_dir / "feature_snapshot.csv",
        "model_predictions.csv": output_dir / "model_predictions.csv",
        "model_metrics.csv": output_dir / "model_metrics.csv",
        "model_confusion_matrix.csv": output_dir / "model_confusion_matrix.csv",
        "model_coefficients.csv": output_dir / "model_coefficients.csv",
        "threshold_selection.csv": output_dir / "threshold_selection.csv",
        "cv_results.csv": output_dir / "cv_results.csv",
        "model_subgroup_metrics.csv": output_dir / "model_subgroup_metrics.csv",
        "model_curve_points.csv": output_dir / "model_curve_points.csv",
        "model_calibration.csv": output_dir / "model_calibration.csv",
        "model_confidence_intervals.csv": output_dir / "model_confidence_intervals.csv",
    }
    snapshot.to_csv(paths["feature_snapshot.csv"], index=False)
    predictions.to_csv(paths["model_predictions.csv"], index=False)
    metrics.to_csv(paths["model_metrics.csv"], index=False)
    confusion.to_csv(paths["model_confusion_matrix.csv"], index=False)
    coefficients.to_csv(paths["model_coefficients.csv"], index=False)
    threshold_table.to_csv(paths["threshold_selection.csv"], index=False)
    cv_results.to_csv(paths["cv_results.csv"], index=False)
    subgroup_metrics.to_csv(paths["model_subgroup_metrics.csv"], index=False)
    curve_points.to_csv(paths["model_curve_points.csv"], index=False)
    calibration.to_csv(paths["model_calibration.csv"], index=False)
    confidence_intervals.to_csv(paths["model_confidence_intervals.csv"], index=False)

    bundle = {
        "estimator": estimator,
        "threshold": threshold,
        "cutoff_day": args.cutoff,
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "categorical_features": MODEL_CATEGORICAL_FEATURES,
        "numeric_features": MODEL_NUMERIC_FEATURES,
        "interaction_features": MODEL_INTERACTION_FEATURES,
        "random_state": args.random_state,
    }
    joblib.dump(bundle, model_path)
    output_hashes = {name: sha256_file(path) for name, path in paths.items()}
    output_hashes[model_path.name] = sha256_file(model_path)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "cutoff_day": args.cutoff,
        "target": "Academic_Fail: Fail=1; Pass/Distinction=0; Withdrawn excluded",
        "grain": ATTEMPT_KEY,
        "best_params": search.best_params_,
        "best_cv_pr_auc": float(search.best_score_),
        "threshold": threshold,
        "threshold_policy": (
            f"maximize validation accuracy subject to recall >= {args.minimum_recall:.2f}"
        ),
        "confidence_interval_policy": (
            "2000 bootstrap iterations resampling id_student groups on final test"
        ),
        "verification_thresholds": VERIFICATION_THRESHOLDS,
        "cohort_counts": cohort_counts,
        "categorical_features": MODEL_CATEGORICAL_FEATURES,
        "numeric_features": MODEL_NUMERIC_FEATURES,
        "interaction_features": MODEL_INTERACTION_FEATURES,
        "audit_only_columns": AUDIT_COLUMNS,
        "prohibited_features": sorted(PROHIBITED_MODEL_FEATURES),
        "output_sha256": output_hashes,
    }
    metadata_path = output_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    output_hashes[metadata_path.name] = sha256_file(metadata_path)

    write_model_report(
        report_path,
        args.cutoff,
        cohort_counts,
        split_summary,
        search.best_params_,
        threshold,
        args.minimum_recall,
        metrics,
        confidence_intervals,
        coefficients,
        output_hashes,
    )

    test_metrics = metrics.loc[
        metrics["model_name"].eq(MODEL_NAME) & metrics["dataset_split"].eq("test")
    ].iloc[0]
    print(f"Model version: {model_version}")
    print(f"Best params: {search.best_params_}")
    print(f"Selected threshold: {threshold:.3f}")
    print(
        "Test metrics: "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"precision={test_metrics['precision_at_risk']:.4f}, "
        f"recall={test_metrics['recall_at_risk']:.4f}, "
        f"f1={test_metrics['f1_at_risk']:.4f}, "
        f"roc_auc={test_metrics['roc_auc']:.4f}, "
        f"pr_auc={test_metrics['pr_auc']:.4f}"
    )
    print(f"Artifacts written to {output_dir}")
    print(f"Model bundle written to {model_path}")
    print(f"Evaluation report written to {report_path}")


def run_validation(args: argparse.Namespace) -> None:
    """Recompute critical invariants and test metrics from exported artifacts."""
    output_dir = Path(args.output_dir).resolve()
    predictions = pd.read_csv(output_dir / "model_predictions.csv")
    metrics = pd.read_csv(output_dir / "model_metrics.csv")
    confusion = pd.read_csv(output_dir / "model_confusion_matrix.csv")
    confidence = pd.read_csv(output_dir / "model_confidence_intervals.csv")
    subgroup = pd.read_csv(output_dir / "model_subgroup_metrics.csv")
    metadata = json.loads((output_dir / "model_metadata.json").read_text(encoding="utf-8"))

    required_columns = {
        *ATTEMPT_KEY,
        "actual_at_risk",
        "predicted_at_risk",
        "risk_probability",
        "actual_fail",
        "predicted_fail",
        "failure_probability",
        "risk_band",
        "dataset_split",
        "error_type",
        "cutoff_day",
        "prediction_threshold",
        "model_name",
        "model_version",
    }
    missing = sorted(required_columns - set(predictions.columns))
    if missing:
        raise AssertionError(f"Prediction output is missing columns: {missing}")
    if predictions[ATTEMPT_KEY].duplicated().any():
        raise AssertionError("Prediction output contains duplicate learner-attempt keys.")
    if not predictions["risk_probability"].between(0, 1).all():
        raise AssertionError("risk_probability contains values outside [0, 1].")

    threshold_values = predictions["prediction_threshold"].unique()
    if len(threshold_values) != 1:
        raise AssertionError("Prediction output has more than one threshold.")
    threshold = float(threshold_values[0])
    recomputed_prediction = predictions["risk_probability"].ge(threshold).astype(int)
    if not recomputed_prediction.equals(predictions["predicted_at_risk"].astype(int)):
        raise AssertionError("predicted_at_risk does not match risk_probability and threshold.")
    recomputed_error = make_error_type(
        predictions["actual_at_risk"].to_numpy(),
        predictions["predicted_at_risk"].to_numpy(),
    )
    if not np.array_equal(recomputed_error, predictions["error_type"].to_numpy()):
        raise AssertionError("error_type is inconsistent with actual/predicted labels.")

    group_sets = {
        split: set(
            predictions.loc[predictions["dataset_split"].eq(split), "id_student"].astype(int)
        )
        for split in ("train", "validation", "test")
    }
    if (
        group_sets["train"] & group_sets["validation"]
        or group_sets["train"] & group_sets["test"]
        or group_sets["validation"] & group_sets["test"]
    ):
        raise AssertionError("id_student appears in more than one exported split.")

    test = predictions.loc[predictions["dataset_split"].eq("test")]
    recomputed = evaluate_predictions(
        test["actual_at_risk"].to_numpy(),
        test["risk_probability"].to_numpy(),
        threshold,
    )
    recorded = metrics.loc[
        metrics["model_name"].eq(MODEL_NAME) & metrics["dataset_split"].eq("test")
    ]
    if len(recorded) != 1:
        raise AssertionError("Expected exactly one Logistic Regression test metric row.")
    recorded_row = recorded.iloc[0]
    for metric, value in recomputed.items():
        if metric in {"rows", "false_negatives", "false_positives"}:
            if int(recorded_row[metric]) != int(value):
                raise AssertionError(f"Metric mismatch for {metric}.")
        elif not np.isclose(float(recorded_row[metric]), float(value), atol=1e-12):
            raise AssertionError(f"Metric mismatch for {metric}.")

    test_confusion = confusion.loc[
        confusion["model_name"].eq(MODEL_NAME)
        & confusion["dataset_split"].eq("test")
    ]
    if int(test_confusion["count"].sum()) != len(test):
        raise AssertionError("Confusion matrix count does not equal test row count.")
    if not np.isclose(float(metadata["threshold"]), threshold, atol=1e-12):
        raise AssertionError("Metadata threshold does not match prediction output.")

    if set(confidence["metric"]) != set(CONFIDENCE_METRICS):
        raise AssertionError("Confidence interval output has unexpected metrics.")
    if not confidence["bootstrap_unit"].eq("id_student").all():
        raise AssertionError("Confidence intervals must resample id_student groups.")
    for row in confidence.itertuples(index=False):
        if not row.lower_95 <= row.point_estimate <= row.upper_95:
            raise AssertionError(f"Invalid confidence interval for {row.metric}.")
        if not np.isclose(
            float(row.point_estimate), float(recorded_row[row.metric]), atol=1e-12
        ):
            raise AssertionError(
                f"Confidence interval point estimate mismatch for {row.metric}."
            )

    checks: list[dict[str, object]] = []

    def add_check(
        name: str,
        observed: object,
        requirement: str,
        passed: bool,
        evidence: str,
    ) -> None:
        checks.append(
            {
                "check": name,
                "observed": observed,
                "requirement": requirement,
                "status": "PASS" if passed else "FAIL",
                "evidence": evidence,
            }
        )

    hash_failures: list[str] = []
    model_path = Path(args.model_path).resolve()
    for name, expected_hash in metadata.get("output_sha256", {}).items():
        artifact = model_path if name.endswith((".joblib", ".pkl")) else output_dir / name
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            hash_failures.append(name)
    add_check(
        "artifact_sha256",
        "all_match" if not hash_failures else ", ".join(hash_failures),
        "mọi artifact phải tồn tại và khớp SHA-256 trong model_metadata.json",
        not hash_failures,
        "model_metadata.json",
    )

    baseline = metrics.loc[
        metrics["model_name"].eq(BASELINE_NAME) & metrics["dataset_split"].eq("test")
    ]
    if len(baseline) != 1:
        raise AssertionError("Expected exactly one baseline test metric row.")
    baseline_row = baseline.iloc[0]
    accuracy_ci = confidence.loc[confidence["metric"].eq("accuracy")].iloc[0]
    presentation_rows = subgroup.loc[subgroup["dimension"].eq("code_presentation")]
    if presentation_rows.empty:
        raise AssertionError("Missing presentation-level subgroup metrics.")

    acceptance_specs = (
        (
            "test_accuracy",
            float(recorded_row["accuracy"]),
            args.minimum_accuracy,
            ">=",
            "model_metrics.csv",
        ),
        (
            "accuracy_ci_lower_95",
            float(accuracy_ci["lower_95"]),
            args.minimum_accuracy_ci_lower,
            ">=",
            "model_confidence_intervals.csv",
        ),
        (
            "test_recall_at_risk",
            float(recorded_row["recall_at_risk"]),
            args.minimum_recall,
            ">=",
            "model_metrics.csv",
        ),
        (
            "test_balanced_accuracy",
            float(recorded_row["balanced_accuracy"]),
            args.minimum_balanced_accuracy,
            ">=",
            "model_metrics.csv",
        ),
        (
            "test_f1_at_risk",
            float(recorded_row["f1_at_risk"]),
            args.minimum_f1,
            ">=",
            "model_metrics.csv",
        ),
        (
            "test_pr_auc_margin_over_prevalence",
            float(recorded_row["pr_auc"] - recorded_row["at_risk_rate"]),
            args.minimum_pr_auc_margin,
            ">=",
            "model_metrics.csv",
        ),
        (
            "test_roc_auc",
            float(recorded_row["roc_auc"]),
            args.minimum_roc_auc,
            ">=",
            "model_metrics.csv",
        ),
        (
            "test_brier_score",
            float(recorded_row["brier_score"]),
            args.maximum_brier,
            "<=",
            "model_metrics.csv",
        ),
        (
            "accuracy_margin_over_dummy",
            float(recorded_row["accuracy"] - baseline_row["accuracy"]),
            args.minimum_baseline_margin,
            ">=",
            "model_metrics.csv",
        ),
        (
            "minimum_presentation_accuracy",
            float(presentation_rows["accuracy"].min()),
            args.minimum_presentation_accuracy,
            ">=",
            "model_subgroup_metrics.csv",
        ),
    )
    for name, observed, limit, operator, evidence in acceptance_specs:
        passed = observed >= limit if operator == ">=" else observed <= limit
        add_check(name, observed, f"{operator} {limit}", passed, evidence)

    verification = pd.DataFrame(checks)
    verification_path = output_dir / "model_verification.csv"
    verification.to_csv(verification_path, index=False)
    failed = verification.loc[verification["status"].eq("FAIL")]
    if not failed.empty:
        print(verification.to_string(index=False))
        raise AssertionError(
            "Model verification failed: " + ", ".join(failed["check"].astype(str))
        )

    print("MODEL_OUTPUT_VALIDATION=PASS")
    print(f"rows={len(predictions):,}; test_rows={len(test):,}; threshold={threshold:.3f}")
    print(
        f"test_accuracy={recomputed['accuracy']:.4f}; "
        f"test_recall={recomputed['recall_at_risk']:.4f}; "
        f"test_f1={recomputed['f1_at_risk']:.4f}; "
        f"test_pr_auc={recomputed['pr_auc']:.4f}"
    )
    print(f"verification_checks={len(verification)}; report={verification_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-safe OULAD Academic-Fail Logistic Regression pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser(
        "audit-cutoffs", help="Compare event coverage for candidate cutoff days."
    )
    audit_parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    audit_parser.add_argument(
        "--cutoffs", nargs="+", type=int, default=[30, 60, 90, 105]
    )
    audit_parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT / "cutoff_audit.csv")
    )
    audit_parser.add_argument("--chunksize", type=int, default=500_000)
    audit_parser.set_defaults(func=run_cutoff_audit)

    train_parser = subparsers.add_parser(
        "train", help="Build cutoff features, tune Logistic Regression and export QA artifacts."
    )
    train_parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    train_parser.add_argument("--cutoff", type=int, required=True)
    train_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    train_parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    train_parser.add_argument("--report", default=str(DEFAULT_REPORT))
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--minimum-recall", type=float, default=0.75)
    train_parser.add_argument("--chunksize", type=int, default=500_000)
    train_parser.add_argument("--n-jobs", type=int, default=-1)
    train_parser.set_defaults(func=run_training)

    validate_parser = subparsers.add_parser(
        "validate", help="Recompute exported model invariants and test metrics."
    )
    validate_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    validate_parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    validate_parser.add_argument(
        "--minimum-accuracy",
        type=float,
        default=VERIFICATION_THRESHOLDS["minimum_accuracy"],
    )
    validate_parser.add_argument(
        "--minimum-accuracy-ci-lower",
        type=float,
        default=VERIFICATION_THRESHOLDS["minimum_accuracy_ci_lower"],
    )
    validate_parser.add_argument(
        "--minimum-recall",
        type=float,
        default=VERIFICATION_THRESHOLDS["minimum_recall"],
    )
    validate_parser.add_argument(
        "--minimum-balanced-accuracy",
        type=float,
        default=VERIFICATION_THRESHOLDS["minimum_balanced_accuracy"],
    )
    validate_parser.add_argument(
        "--minimum-f1", type=float, default=VERIFICATION_THRESHOLDS["minimum_f1"]
    )
    validate_parser.add_argument(
        "--minimum-pr-auc-margin",
        type=float,
        default=VERIFICATION_THRESHOLDS["minimum_pr_auc_margin"],
    )
    validate_parser.add_argument(
        "--minimum-roc-auc",
        type=float,
        default=VERIFICATION_THRESHOLDS["minimum_roc_auc"],
    )
    validate_parser.add_argument(
        "--maximum-brier",
        type=float,
        default=VERIFICATION_THRESHOLDS["maximum_brier"],
    )
    validate_parser.add_argument(
        "--minimum-baseline-margin",
        type=float,
        default=VERIFICATION_THRESHOLDS["minimum_baseline_margin"],
    )
    validate_parser.add_argument(
        "--minimum-presentation-accuracy",
        type=float,
        default=VERIFICATION_THRESHOLDS["minimum_presentation_accuracy"],
    )
    validate_parser.set_defaults(func=run_validation)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bounded_arguments = (
        "minimum_accuracy",
        "minimum_accuracy_ci_lower",
        "minimum_recall",
        "minimum_balanced_accuracy",
        "minimum_f1",
        "minimum_pr_auc_margin",
        "minimum_roc_auc",
        "maximum_brier",
        "minimum_baseline_margin",
        "minimum_presentation_accuracy",
    )
    for name in bounded_arguments:
        if hasattr(args, name) and not 0 <= getattr(args, name) <= 1:
            option = "--" + name.replace("_", "-")
            raise ValueError(f"{option} must be in [0, 1].")
    args.func(args)


if __name__ == "__main__":
    main()
