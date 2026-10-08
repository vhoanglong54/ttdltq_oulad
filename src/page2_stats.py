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
    
    # Target is At_Risk (Fail only)
    df105["At_Risk"] = df105["final_result"].eq("Fail").astype(int)
    
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
    
    # Only keep Pass, Distinction, Fail (drop Withdrawn for model evaluation)
    d_model = d_clean[d_clean["final_result"].isin(["Pass", "Distinction", "Fail"])].copy()
    
    rows = []
    for f in feats:
        a = roc_auc_score(d_model["At_Risk"], d_model[f])
        rows.append({"yeu_to": f, "AUC": max(a, 1-a), "huong": "tăng -> rủi ro tăng" if a > 0.5 else "tăng -> rủi ro giảm"})
    rank = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    rank.to_csv(OUTPUT_DIR / "auc_ranking.csv", index=False)
    
    # 3. Actionable Rule metrics
    q1 = d["total_clicks"].quantile(0.25)
    
    # Evaluate rule on the cohort (df105 without early_wd)
    d_rule = d[d["final_result"].isin(["Pass", "Distinction", "Fail"])].copy()
    flag_rule = (d_rule["submit_rate"] == 0) & (d_rule["total_clicks"] <= q1)
    share = flag_rule.mean()
    precision = d_rule.loc[flag_rule, "At_Risk"].mean() if flag_rule.sum() > 0 else 0
    recall = d_rule.loc[flag_rule, "At_Risk"].sum() / d_rule["At_Risk"].sum() if d_rule["At_Risk"].sum() > 0 else 0
    
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
    # Filter to Pass/Distinction vs Fail
    vle = vle[vle["final_result"].isin(["Pass", "Distinction", "Fail"])].copy()
    vle["At_Risk"] = vle["final_result"].eq("Fail").astype(int)
    per = vle.groupby(["At_Risk", "activity_type"], observed=True)["sum_click"].sum().reset_index()
    
    df_vle = df[df["final_result"].isin(["Pass", "Distinction", "Fail"])].copy()
    df_vle["At_Risk"] = df_vle["final_result"].eq("Fail").astype(int)
    totals = df_vle.groupby("At_Risk")["id_student"].nunique().reset_index(name="students")
    
    per = per.merge(totals, on="At_Risk")
    per["avg_click"] = per["sum_click"] / per["students"]
    
    avg = per.pivot(index="activity_type", columns="At_Risk", values="avg_click").fillna(0)
    # Add small epsilon to avoid divide by zero
    avg["ratio"] = avg[0] / (avg[1] + 1e-6)
    avg.reset_index().to_csv(OUTPUT_DIR / "resource_type_ratio.csv", index=False)
    
if __name__ == "__main__":
    build_page2_stats()
    print("Page 2 stats computed.")
