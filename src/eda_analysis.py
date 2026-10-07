"""Reproducible EDA and insight evidence for the approved OULAD plan.

The script deliberately separates descriptive analysis from model evaluation:

* full-course `clean_dataset.csv` is used for outcome and regional summaries;
* the day-105 academic-fail snapshot is used for mid-course signal analysis;
* model metrics are never written as analytical insights.

Run from the repository root:

    python src/eda_analysis.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "processed" / "clean_dataset.csv"
SNAPSHOT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "model_academic_fail"
    / "c105_final"
    / "feature_snapshot.csv"
)
VLE_PATH = ROOT / "data" / "interim" / "studentVle.csv"
TABLE_DIR = ROOT / "reports" / "eda"
FIGURE_DIR = ROOT / "reports" / "figures" / "eda"

ATTEMPT_KEY = ["code_module", "code_presentation", "id_student"]
RESULT_ORDER = ["Distinction", "Pass", "Fail", "Withdrawn"]
RISK_LABELS = {0: "Pass/Distinction", 1: "Fail"}
QUARTILE_ORDER = ["Q1 — Lowest", "Q2", "Q3", "Q4 — Highest"]
ASSESSMENT_SCORE_ORDER = QUARTILE_ORDER + ["No scored assessment by cutoff"]
COMPLETION_ORDER = ["0%", "(0%, 50%]", "(50%, 100%)", "100%"]
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class InsightEvidence:
    insight_id: str
    rq: str
    comparison: str
    group_a: str
    group_a_n: int
    group_a_rate: float
    group_b: str
    group_b_n: int
    group_b_rate: float
    risk_difference_pp: float
    risk_ratio: float
    scope: str
    limitation: str


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input: {path}. Run the data/model pipeline first."
        )


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return a 95% Wilson interval for a binomial proportion."""

    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1 + (Z_95**2 / total)
    centre = (proportion + Z_95**2 / (2 * total)) / denominator
    margin = (
        Z_95
        * math.sqrt(
            proportion * (1 - proportion) / total + Z_95**2 / (4 * total**2)
        )
        / denominator
    )
    return centre - margin, centre + margin


def summarize_risk(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    summary = (
        frame.groupby(groups, observed=True, dropna=False)
        .agg(attempts=("At_Risk", "size"), at_risk_count=("At_Risk", "sum"))
        .reset_index()
    )
    summary["at_risk_rate"] = summary["at_risk_count"] / summary["attempts"]
    intervals = [
        wilson_interval(int(successes), int(total))
        for successes, total in zip(
            summary["at_risk_count"], summary["attempts"], strict=True
        )
    ]
    summary["ci95_low"] = [item[0] for item in intervals]
    summary["ci95_high"] = [item[1] for item in intervals]
    return summary


def add_analysis_bands(snapshot: pd.DataFrame) -> pd.DataFrame:
    frame = snapshot.copy()
    frame["actual_status"] = frame["At_Risk"].map(RISK_LABELS)

    for source, target in (
        ("vle_total_clicks_cutoff", "engagement_quartile"),
        ("vle_active_days_cutoff", "active_days_quartile"),
        ("vle_active_days_last_28_days", "recent_active_days_quartile"),
    ):
        frame[target] = pd.qcut(
            frame[source].rank(method="first"),
            q=4,
            labels=QUARTILE_ORDER,
        )

    # A missing assessment score is not silently mixed into a numeric quartile:
    # it normally means that no scored assessment was available/completed by
    # the cutoff. Keep that analytically distinct and let completion carry the
    # corresponding behavioural signal.
    score = frame["assessment_weighted_score_cutoff"]
    has_score = score.notna()
    score_band = pd.Series(
        "No scored assessment by cutoff", index=frame.index, dtype="object"
    )
    score_band.loc[has_score] = pd.qcut(
        score.loc[has_score].rank(method="first"),
        q=4,
        labels=QUARTILE_ORDER,
    ).astype(str)
    frame["assessment_score_quartile"] = pd.Categorical(
        score_band,
        categories=ASSESSMENT_SCORE_ORDER,
        ordered=True,
    )

    completion = frame["assessment_completion_rate_cutoff"]
    frame["assessment_completion_band"] = np.select(
        [
            completion.eq(0),
            completion.gt(0) & completion.le(0.5),
            completion.gt(0.5) & completion.lt(1),
            completion.eq(1),
        ],
        COMPLETION_ORDER,
        default="Unexpected",
    )
    frame["assessment_completion_band"] = pd.Categorical(
        frame["assessment_completion_band"],
        categories=COMPLETION_ORDER,
        ordered=True,
    )
    frame["previous_attempt_band"] = np.where(
        frame["num_of_prev_attempts"].eq(0), "0 previous attempts", "1+ previous attempts"
    )
    delay = frame["assessment_late_days_mean_cutoff"]
    has_delay = delay.notna()
    delay_band = pd.Series("No submitted assessment", index=frame.index, dtype="object")
    delay_band.loc[has_delay] = pd.qcut(
        delay.loc[has_delay].rank(method="first"),
        q=4,
        labels=QUARTILE_ORDER,
    ).astype(str)
    frame["submission_delay_quartile"] = delay_band
    frame["engagement_assessment_profile"] = (
        frame["engagement_quartile"].astype(str)
        + " | "
        + frame["assessment_score_quartile"].astype(str)
    )
    return frame


def build_weekly_vle_trend(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Aggregate full weeks 0–14 before the day-105 cutoff.

    The denominator is every eligible attempt in each outcome group, including
    attempts with zero clicks in a week. Day 105 is excluded because it would be
    a one-day partial week and would create a misleading visual drop.
    """

    eligible = snapshot[ATTEMPT_KEY + ["At_Risk"]].copy()
    denominators = eligible.groupby("At_Risk", observed=True).size().to_dict()
    click_parts: list[pd.DataFrame] = []
    active_parts: list[pd.DataFrame] = []
    usecols = ATTEMPT_KEY + ["date", "sum_click"]

    for chunk in pd.read_csv(VLE_PATH, usecols=usecols, chunksize=600_000):
        chunk = chunk.loc[chunk["date"].between(0, 104)]
        if chunk.empty:
            continue
        chunk = chunk.merge(eligible, on=ATTEMPT_KEY, how="inner", validate="many_to_one")
        chunk["week"] = (chunk["date"] // 7).astype(int)
        click_parts.append(
            chunk.groupby(["week", "At_Risk"], observed=True, as_index=False).agg(
                total_clicks=("sum_click", "sum")
            )
        )
        active_parts.append(
            chunk[ATTEMPT_KEY + ["week", "At_Risk"]].drop_duplicates()
        )

    clicks = (
        pd.concat(click_parts, ignore_index=True)
        .groupby(["week", "At_Risk"], observed=True, as_index=False)["total_clicks"]
        .sum()
    )
    active = pd.concat(active_parts, ignore_index=True).drop_duplicates(
        ATTEMPT_KEY + ["week", "At_Risk"]
    )
    active = (
        active.groupby(["week", "At_Risk"], observed=True)
        .size()
        .rename("active_attempts")
        .reset_index()
    )
    trend = clicks.merge(active, on=["week", "At_Risk"], how="outer").fillna(0)
    trend["eligible_attempts"] = trend["At_Risk"].map(denominators).astype(int)
    trend["mean_clicks_per_attempt"] = (
        trend["total_clicks"] / trend["eligible_attempts"]
    )
    trend["active_attempt_rate"] = trend["active_attempts"] / trend["eligible_attempts"]
    trend["actual_status"] = trend["At_Risk"].map(RISK_LABELS)
    trend["day_start"] = trend["week"] * 7
    trend["day_end"] = trend["day_start"] + 6
    return trend.sort_values(["week", "At_Risk"]).reset_index(drop=True)


def compare_two_groups(
    frame: pd.DataFrame,
    column: str,
    group_a: str,
    group_b: str,
    *,
    insight_id: str,
    rq: str,
    comparison: str,
    scope: str,
    limitation: str,
) -> InsightEvidence:
    grouped = frame.groupby(column, observed=True)["At_Risk"].agg(["size", "mean"])
    a_n, a_rate = int(grouped.loc[group_a, "size"]), float(grouped.loc[group_a, "mean"])
    b_n, b_rate = int(grouped.loc[group_b, "size"]), float(grouped.loc[group_b, "mean"])
    return InsightEvidence(
        insight_id=insight_id,
        rq=rq,
        comparison=comparison,
        group_a=group_a,
        group_a_n=a_n,
        group_a_rate=a_rate,
        group_b=group_b,
        group_b_n=b_n,
        group_b_rate=b_rate,
        risk_difference_pp=(a_rate - b_rate) * 100,
        risk_ratio=a_rate / b_rate if b_rate else float("nan"),
        scope=scope,
        limitation=limitation,
    )


def build_insight_evidence(
    clean: pd.DataFrame,
    snapshot: pd.DataFrame,
    module_presentation: pd.DataFrame,
    region: pd.DataFrame,
) -> pd.DataFrame:
    evidence: list[InsightEvidence] = []

    lowest_mp = module_presentation.sort_values("at_risk_rate").iloc[0]
    highest_mp = module_presentation.sort_values("at_risk_rate").iloc[-1]
    evidence.append(
        InsightEvidence(
            insight_id="INS-01",
            rq="RQ1",
            comparison="Module-presentation có tỷ lệ Fail cao nhất so với thấp nhất",
            group_a=f"{highest_mp.code_module}-{highest_mp.code_presentation}",
            group_a_n=int(highest_mp.attempts),
            group_a_rate=float(highest_mp.at_risk_rate),
            group_b=f"{lowest_mp.code_module}-{lowest_mp.code_presentation}",
            group_b_n=int(lowest_mp.attempts),
            group_b_rate=float(lowest_mp.at_risk_rate),
            risk_difference_pp=(
                float(highest_mp.at_risk_rate) - float(lowest_mp.at_risk_rate)
            )
            * 100,
            risk_ratio=float(highest_mp.at_risk_rate) / float(lowest_mp.at_risk_rate),
            scope=f"Full descriptive table; N={len(clean):,} attempts",
            limitation="Khác biệt có thể phản ánh cấu trúc module, assessment và cohort; không phải tác động nhân quả.",
        )
    )
    evidence.append(
        compare_two_groups(
            snapshot,
            "engagement_quartile",
            "Q1 — Lowest",
            "Q4 — Highest",
            insight_id="INS-02",
            rq="RQ2",
            comparison="Tỷ lệ Fail giữa quartile VLE clicks thấp nhất và cao nhất",
            scope=f"Eligible cutoff-day-105 snapshot; N={len(snapshot):,} attempts",
            limitation="Clicks đo tương tác nền tảng, không đo chất lượng học hoặc thời gian học.",
        )
    )
    evidence.append(
        compare_two_groups(
            snapshot,
            "assessment_completion_band",
            "0%",
            "100%",
            insight_id="INS-03",
            rq="RQ3",
            comparison="Tỷ lệ Fail giữa 0% và 100% assessment completion trước cutoff",
            scope=f"Eligible cutoff-day-105 snapshot; N={len(snapshot):,} attempts",
            limitation="Assessment schedule khác theo module; completion phản ánh trạng thái đến cutoff, không phải nguyên nhân duy nhất.",
        )
    )

    low_low = snapshot[
        snapshot["engagement_quartile"].eq("Q1 — Lowest")
        & snapshot["assessment_score_quartile"].eq("Q1 — Lowest")
    ]
    high_high = snapshot[
        snapshot["engagement_quartile"].eq("Q4 — Highest")
        & snapshot["assessment_score_quartile"].eq("Q4 — Highest")
    ]
    low_rate, high_rate = float(low_low.At_Risk.mean()), float(high_high.At_Risk.mean())
    evidence.append(
        InsightEvidence(
            insight_id="INS-04",
            rq="RQ5",
            comparison="Tương tác giữa engagement quartile và assessment-score quartile",
            group_a="Low engagement + low assessment",
            group_a_n=len(low_low),
            group_a_rate=low_rate,
            group_b="High engagement + high assessment",
            group_b_n=len(high_high),
            group_b_rate=high_rate,
            risk_difference_pp=(low_rate - high_rate) * 100,
            risk_ratio=low_rate / high_rate if high_rate else float("nan"),
            scope=f"Eligible cutoff-day-105 snapshot; N={len(snapshot):,} attempts",
            limitation="Nhóm quartile là mô tả tương tác, không chứng minh hiệu ứng nhân quả hoặc ngưỡng can thiệp.",
        )
    )
    evidence.append(
        compare_two_groups(
            snapshot,
            "previous_attempt_band",
            "1+ previous attempts",
            "0 previous attempts",
            insight_id="INS-05",
            rq="RQ4",
            comparison="At-Risk rate theo lịch sử số lần học module",
            scope=f"Eligible cutoff-day-105 snapshot; N={len(snapshot):,} attempts",
            limitation="Không có kết quả chi tiết của lần học trước và không kiểm soát đầy đủ khác biệt module/cohort.",
        )
    )

    lowest_region = region.sort_values("at_risk_rate").iloc[0]
    highest_region = region.sort_values("at_risk_rate").iloc[-1]
    evidence.append(
        InsightEvidence(
            insight_id="INS-06",
            rq="RQ4",
            comparison="Region có At-Risk rate cao nhất so với thấp nhất",
            group_a=str(highest_region.region),
            group_a_n=int(highest_region.attempts),
            group_a_rate=float(highest_region.at_risk_rate),
            group_b=str(lowest_region.region),
            group_b_n=int(lowest_region.attempts),
            group_b_rate=float(lowest_region.at_risk_rate),
            risk_difference_pp=(
                float(highest_region.at_risk_rate) - float(lowest_region.at_risk_rate)
            )
            * 100,
            risk_ratio=float(highest_region.at_risk_rate) / float(lowest_region.at_risk_rate),
            scope=f"Full descriptive table; N={len(clean):,} attempts",
            limitation="OULAD dùng vùng hành chính lịch sử của Open University; region và IMD không đại diện nguyên nhân cá nhân.",
        )
    )
    evidence.append(
        compare_two_groups(
            snapshot.loc[snapshot["submission_delay_quartile"].ne("No submitted assessment")],
            "submission_delay_quartile",
            "Q4 — Highest",
            "Q1 — Lowest",
            insight_id="INS-07",
            rq="RQ2",
            comparison="Tỷ lệ Fail giữa quartile độ trễ nộp bài cao nhất và thấp nhất",
            scope=f"Eligible day-105 attempts with submitted assessments; N={snapshot['assessment_late_days_mean_cutoff'].notna().sum():,}",
            limitation="Độ trễ trung bình chỉ tính bài đã nộp; nhóm không nộp được phản ánh riêng qua completion/missed due count.",
        )
    )
    evidence.append(
        compare_two_groups(
            snapshot,
            "recent_active_days_quartile",
            "Q1 — Lowest",
            "Q4 — Highest",
            insight_id="INS-08",
            rq="RQ2",
            comparison="Tỷ lệ Fail giữa quartile số ngày hoạt động VLE 28 ngày thấp nhất và cao nhất",
            scope=f"Eligible cutoff-day-105 snapshot; N={len(snapshot):,} attempts",
            limitation="Hoạt động VLE không bao quát việc học offline hoặc chất lượng tương tác.",
        )
    )
    return pd.DataFrame([item.__dict__ for item in evidence])


def build_hypothesis_results(
    clean: pd.DataFrame,
    snapshot: pd.DataFrame,
    weekly: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    by_id = evidence.set_index("insight_id")
    active = summarize_risk(snapshot, ["active_days_quartile"])
    score = summarize_risk(snapshot, ["assessment_score_quartile"])

    trend_pivot = weekly.pivot(
        index="week", columns="At_Risk", values="mean_clicks_per_attempt"
    )
    weeks_at_risk_lower = int((trend_pivot[1] < trend_pivot[0]).sum())
    total_weeks = len(trend_pivot)

    module_effects: list[float] = []
    for _, module_rows in snapshot.groupby("code_module", observed=True):
        local = module_rows.copy()
        local["quartile"] = pd.qcut(
            local["vle_total_clicks_cutoff"].rank(method="first"),
            q=4,
            labels=QUARTILE_ORDER,
        )
        rates = local.groupby("quartile", observed=True).At_Risk.mean()
        module_effects.append(float(rates.loc["Q1 — Lowest"] - rates.loc["Q4 — Highest"]))

    return pd.DataFrame(
        [
            {
                "hypothesis": "H01",
                "status": "Ủng hộ ở mức mô tả",
                "evidence": f"Module-presentation range={by_id.loc['INS-01','risk_difference_pp']:.2f} pp",
                "scope": "Full descriptive table",
            },
            {
                "hypothesis": "H02",
                "status": "Ủng hộ ở mức mô tả",
                "evidence": f"Lowest vs highest clicks quartile difference={by_id.loc['INS-02','risk_difference_pp']:.2f} pp",
                "scope": "Day-105 snapshot",
            },
            {
                "hypothesis": "H03",
                "status": "Ủng hộ ở mức mô tả",
                "evidence": f"Lowest vs highest active-days quartile difference={(active.iloc[0].at_risk_rate-active.iloc[-1].at_risk_rate)*100:.2f} pp",
                "scope": "Day-105 snapshot",
            },
            {
                "hypothesis": "H04",
                "status": "Ủng hộ ở mức mô tả",
                "evidence": f"Fail-group mean weekly clicks lower in {weeks_at_risk_lower}/{total_weeks} complete weeks",
                "scope": "Weeks 0–14 among day-105 eligible attempts",
            },
            {
                "hypothesis": "H05",
                "status": "Ủng hộ ở mức mô tả",
                "evidence": f"0% vs 100% completion difference={by_id.loc['INS-03','risk_difference_pp']:.2f} pp",
                "scope": "Day-105 snapshot",
            },
            {
                "hypothesis": "H06",
                "status": "Ủng hộ ở mức mô tả",
                "evidence": f"Lowest vs highest assessment-score quartile difference={(score.set_index('assessment_score_quartile').loc['Q1 — Lowest','at_risk_rate']-score.set_index('assessment_score_quartile').loc['Q4 — Highest','at_risk_rate'])*100:.2f} pp",
                "scope": "Day-105 snapshot",
            },
            {
                "hypothesis": "H07",
                "status": "Ủng hộ ở mức mô tả",
                "evidence": f"1+ vs 0 previous-attempt difference={by_id.loc['INS-05','risk_difference_pp']:.2f} pp",
                "scope": "Day-105 snapshot",
            },
            {
                "hypothesis": "H08",
                "status": "Ủng hộ ở mức mô tả; cần nêu confounding",
                "evidence": f"Regional range={by_id.loc['INS-06','risk_difference_pp']:.2f} pp",
                "scope": "Full descriptive table",
            },
            {
                "hypothesis": "H09",
                "status": "Ủng hộ ở mức mô tả",
                "evidence": f"Low-low vs high-high difference={by_id.loc['INS-04','risk_difference_pp']:.2f} pp",
                "scope": "Day-105 snapshot",
            },
            {
                "hypothesis": "H10",
                "status": "Ủng hộ: độ lớn khác theo module",
                "evidence": f"Within-module low-vs-high engagement difference range={min(module_effects)*100:.2f}–{max(module_effects)*100:.2f} pp",
                "scope": "Day-105 snapshot, quartiles calculated within module",
            },
        ]
    )


def save_tables(clean: pd.DataFrame, snapshot: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outcome = (
        clean.groupby(["code_module", "code_presentation", "final_result"], observed=True)
        .size()
        .rename("attempts")
        .reset_index()
    )
    module_presentation = summarize_risk(
        clean, ["code_module", "code_presentation"]
    )
    region = summarize_risk(clean, ["region"])
    engagement = summarize_risk(snapshot, ["engagement_quartile"])
    active_days = summarize_risk(snapshot, ["active_days_quartile"])
    recent_active_days = summarize_risk(snapshot, ["recent_active_days_quartile"])
    completion = summarize_risk(snapshot, ["assessment_completion_band"])
    assessment_score = summarize_risk(snapshot, ["assessment_score_quartile"])
    interaction = summarize_risk(
        snapshot, ["engagement_quartile", "assessment_score_quartile"]
    )
    previous_attempts = summarize_risk(snapshot, ["previous_attempt_band"])
    submission_delay = summarize_risk(snapshot, ["submission_delay_quartile"])
    education = summarize_risk(snapshot, ["highest_education"])
    weekly = build_weekly_vle_trend(snapshot)
    evidence = build_insight_evidence(clean, snapshot, module_presentation, region)
    hypotheses = build_hypothesis_results(clean, snapshot, weekly, evidence)

    tables = {
        "outcome_by_module_presentation": outcome,
        "module_presentation_risk": module_presentation,
        "region_risk": region,
        "engagement_quartiles": engagement,
        "active_days_quartiles": active_days,
        "recent_active_days_quartiles": recent_active_days,
        "assessment_completion": completion,
        "assessment_score_quartiles": assessment_score,
        "engagement_assessment_matrix": interaction,
        "previous_attempts": previous_attempts,
        "submission_delay_quartiles": submission_delay,
        "education_risk": education,
        "vle_weekly_trend": weekly,
        "insight_evidence": evidence,
        "hypothesis_results": hypotheses,
    }
    for name, table in tables.items():
        table.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    return tables


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_static_evidence(
    clean: pd.DataFrame,
    snapshot: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> None:
    configure_plotting()

    outcome = tables["outcome_by_module_presentation"].copy()
    outcome["module_presentation"] = (
        outcome["code_module"] + "-" + outcome["code_presentation"]
    )
    pivot = (
        outcome.pivot(
            index="module_presentation", columns="final_result", values="attempts"
        )
        .fillna(0)
        .reindex(columns=RESULT_ORDER, fill_value=0)
    )
    proportions = pivot.div(pivot.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(13, 7))
    proportions.plot(
        kind="bar",
        stacked=True,
        color=["#0F766E", "#2563EB", "#F97316", "#DC2626"],
        ax=ax,
    )
    ax.set_title("Final-result distribution by module-presentation")
    ax.set_xlabel("Module-presentation")
    ax.set_ylabel("Share of learning attempts")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(title="Final result", bbox_to_anchor=(1.01, 1), loc="upper left")
    save_figure(fig, "01_outcome_by_module_presentation.png")

    weekly = tables["vle_weekly_trend"]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(
        data=weekly,
        x="week",
        y="mean_clicks_per_attempt",
        hue="actual_status",
        marker="o",
        palette={"Pass/Distinction": "#2563EB", "Fail": "#DC2626"},
        ax=ax,
    )
    ax.set_title("Mean weekly VLE clicks per eligible attempt (days 0–104)")
    ax.set_xlabel("Complete week since module start")
    ax.set_ylabel("Mean clicks per eligible attempt")
    save_figure(fig, "02_vle_weekly_trend.png")

    sample = snapshot.sample(min(len(snapshot), 12_000), random_state=42).copy()
    sample["log1p_vle_clicks"] = np.log1p(sample["vle_total_clicks_cutoff"])
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(
        data=sample,
        x="actual_status",
        y="log1p_vle_clicks",
        hue="actual_status",
        palette={"Pass/Distinction": "#2563EB", "Fail": "#DC2626"},
        legend=False,
        ax=ax,
    )
    ax.set_title("VLE engagement through day 105 by academic result")
    ax.set_xlabel("Actual final status")
    ax.set_ylabel("log(1 + VLE clicks through day 105)")
    save_figure(fig, "03_early_vle_boxplot.png")

    completion = tables["assessment_completion"].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=completion,
        x="assessment_completion_band",
        y="at_risk_rate",
        color="#F97316",
        ax=ax,
    )
    ax.set_title("Fail rate by assessment completion through day 105")
    ax.set_xlabel("Assessment completion band")
    ax.set_ylabel("Fail rate")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for index, row in completion.reset_index(drop=True).iterrows():
        ax.text(index, row.at_risk_rate + 0.02, f"N={row.attempts:,}", ha="center", fontsize=9)
    save_figure(fig, "04_assessment_completion_risk.png")

    interaction = tables["engagement_assessment_matrix"].copy()
    matrix = interaction.pivot(
        index="engagement_quartile",
        columns="assessment_score_quartile",
        values="at_risk_rate",
    ).reindex(index=QUARTILE_ORDER, columns=QUARTILE_ORDER)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1%",
        cmap="OrRd",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Fail rate"},
        ax=ax,
    )
    ax.set_title("Interaction of early VLE engagement and assessment score")
    ax.set_xlabel("Assessment-score quartile")
    ax.set_ylabel("VLE-click quartile")
    save_figure(fig, "05_engagement_assessment_heatmap.png")

    region = tables["region_risk"].sort_values("at_risk_rate")
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.barplot(data=region, x="at_risk_rate", y="region", color="#DC2626", ax=ax)
    ax.set_title("Fail rate by OULAD region")
    ax.set_xlabel("Fail rate")
    ax.set_ylabel("Region")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for index, row in region.reset_index(drop=True).iterrows():
        ax.text(row.at_risk_rate + 0.004, index, f"N={row.attempts:,}", va="center", fontsize=8)
    save_figure(fig, "06_region_at_risk_rate.png")


def main() -> None:
    for path in (CLEAN_PATH, SNAPSHOT_PATH, VLE_PATH):
        require_file(path)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(CLEAN_PATH)
    clean["Academic_Fail"] = clean["final_result"].eq("Fail").astype("int8")
    clean["Withdrawn_Flag"] = clean["final_result"].eq("Withdrawn").astype("int8")
    clean["At_Risk"] = clean["Academic_Fail"]
    snapshot = pd.read_csv(SNAPSHOT_PATH)
    duplicate_clean = int(clean.duplicated(ATTEMPT_KEY).sum())
    duplicate_snapshot = int(snapshot.duplicated(ATTEMPT_KEY).sum())
    if duplicate_clean or duplicate_snapshot:
        raise ValueError(
            "EDA stopped because attempt keys are not unique: "
            f"clean={duplicate_clean}, snapshot={duplicate_snapshot}."
        )
    snapshot = add_analysis_bands(snapshot)
    tables = save_tables(clean, snapshot)
    plot_static_evidence(clean, snapshot, tables)
    print(
        "EDA complete: "
        f"{len(tables)} tables in {TABLE_DIR.relative_to(ROOT)} and "
        f"6 figures in {FIGURE_DIR.relative_to(ROOT)}."
    )


if __name__ == "__main__":
    main()
