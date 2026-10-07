import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_dataset.csv"
SNAPSHOT_PATH = PROJECT_ROOT / "data" / "processed" / "model_academic_fail" / "c105_final" / "feature_snapshot.csv"
SUBMISSIONS_PATH = PROJECT_ROOT / "data" / "processed" / "dashboard" / "assessment_submissions.csv.gz"
VLE_DAY105_PATH = PROJECT_ROOT / "data" / "processed" / "dashboard" / "vle_activity_summary_day105.csv.gz"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "dashboard"
ATTEMPT_KEY = ["code_module", "code_presentation", "id_student"]

def build_page2_stats():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_DATA_PATH)
    snapshot = pd.read_csv(SNAPSHOT_PATH)
    
    # 1. df105 - keep only people who didn't withdraw early
    df["early_wd"] = df["date_unregistration"].le(105)
    df105 = df[~df["early_wd"]].copy()
    df105["not_complete"] = df105["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    
    # Merge snapshot features into df105
    d = df105.merge(snapshot, on=ATTEMPT_KEY, how="inner", suffixes=("", "_snap"))
    
    # 2. AUC Ranking
    d["submit_rate"] = d["assessment_completion_rate_cutoff"]
    d["total_clicks"] = d["vle_total_clicks_cutoff"]
    d["active_days"] = d["vle_active_days_cutoff"]
    d["clicks_last14"] = d["vle_clicks_last_7_days"] + d["vle_clicks_previous_7_days"]
    d["avg_score"] = d["assessment_weighted_score_cutoff"]
    d["num_of_prev_attempts"] = d["num_of_prev_attempts"]
    d["studied_credits"] = d["studied_credits"]
    
    feats = ["submit_rate", "total_clicks", "active_days", "clicks_last14", "avg_score", "num_of_prev_attempts", "studied_credits"]
    d_clean = d.dropna(subset=feats).copy()
    
    rows = []
    for f in feats:
        a = roc_auc_score(d_clean["not_complete"], d_clean[f])
        rows.append({"yeu_to": f, "AUC": max(a, 1-a), "huong": "tăng -> rủi ro tăng" if a > 0.5 else "tăng -> rủi ro giảm"})
    rank = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    rank.to_csv(OUTPUT_DIR / "auc_ranking.csv", index=False)
    
    # 3. Actionable Rule metrics
    q1 = d["total_clicks"].quantile(0.25)
    flag = (d["submit_rate"] == 0) & (d["total_clicks"] <= q1)
    share = flag.mean()
    precision = d.loc[flag, "not_complete"].mean() if flag.sum() > 0 else 0
    recall = d.loc[flag, "not_complete"].sum() / d["not_complete"].sum() if d["not_complete"].sum() > 0 else 0
    
    pd.DataFrame([{"share": share, "precision": precision, "recall": recall}]).to_csv(OUTPUT_DIR / "rule_metrics.csv", index=False)
    
    # 4. Submissions delay bucket and score
    subs = pd.read_csv(SUBMISSIONS_PATH)
    bins = [-np.inf, -8, -1, 0, 7, np.inf]
    labels = ["Sớm >7 ngày","Sớm 1-7 ngày","Đúng hạn","Trễ 1-7 ngày","Trễ >7 ngày"]
    subs["delay_bucket"] = pd.cut(subs["submission_delay"], bins=bins, labels=labels)
    subs["score_c"] = subs["score"] - subs.groupby("id_assessment")["score"].transform("mean")
    tab = subs.groupby("delay_bucket", observed=True)["score_c"].agg(["median", "count"]).reset_index()
    
    spearman = subs[["submission_delay", "score_c"]].corr(method="spearman").iloc[0, 1]
    tab.to_csv(OUTPUT_DIR / "delay_score_buckets.csv", index=False)
    pd.DataFrame([{"spearman": spearman}]).to_csv(OUTPUT_DIR / "delay_spearman.csv", index=False)
    
    # 5. Resource type ratio
    vle = pd.read_csv(VLE_DAY105_PATH)
    # vle is aggregated by final_result and activity_type
    vle["not_complete"] = vle["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    per = vle.groupby(["not_complete", "activity_type"], observed=True)["sum_click"].sum().reset_index()
    
    # Get students count per group (we use df instead of df105 because VLE data wasn't filtered for early_wd)
    df["not_complete"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    totals = df.groupby("not_complete")["id_student"].nunique().reset_index(name="students")
    
    per = per.merge(totals, on="not_complete")
    per["avg_click"] = per["sum_click"] / per["students"]
    
    avg = per.pivot(index="activity_type", columns="not_complete", values="avg_click").fillna(0)
    avg["ratio"] = avg[0] / avg[1].replace(0, np.nan)
    avg.reset_index().to_csv(OUTPUT_DIR / "resource_type_ratio.csv", index=False)
    
if __name__ == "__main__":
    build_page2_stats()
    print("Page 2 stats computed.")
