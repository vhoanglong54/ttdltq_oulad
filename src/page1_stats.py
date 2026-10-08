import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_dataset.csv"
DEADLINES_PATH = PROJECT_ROOT / "data" / "processed" / "dashboard" / "assessment_deadlines.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "dashboard"
ATTEMPT_KEY = ["code_module", "code_presentation", "id_student"]

def build_no_submission_stats():
    df = pd.read_csv(CLEAN_DATA_PATH)
    # n_sub = 0 if assessment_event_count is 0 or NaN
    df["n_sub"] = df["assessment_event_count"].fillna(0)
    df["no_sub"] = (df["n_sub"] == 0).astype(int)
    
    no_sub_rate = df.groupby("final_result")["no_sub"].agg(["mean", "size"]).reset_index()
    no_sub_rate.rename(columns={"mean": "no_sub_rate", "size": "attempts"}, inplace=True)
    no_sub_rate.to_csv(OUTPUT_DIR / "no_submission_rates.csv", index=False)

def build_unregistration_timeline():
    df = pd.read_csv(CLEAN_DATA_PATH)
    deadlines = pd.read_csv(DEADLINES_PATH)
    
    w = df[(df["final_result"] == "Withdrawn")].dropna(subset=["date_unregistration"]).copy()
    w["week"] = (w["date_unregistration"] // 7).clip(lower=-4)
    
    # cumulative proportion among Withdrawn
    cum = w.groupby("week").size().cumsum() / len(w)
    cum_df = cum.reset_index().rename(columns={0: "cumulative_rate"})
    cum_df.to_csv(OUTPUT_DIR / "withdrawn_timeline.csv", index=False)
    
    first_due = deadlines.dropna(subset=["due_date"]).groupby(["code_module", "code_presentation"])["due_date"].min().rename("first_due").reset_index()
    w = w.merge(first_due, on=["code_module", "code_presentation"], how="left")
    
    share_before_first = (w["date_unregistration"] < w["first_due"]).mean()
    pd.DataFrame([{"share_before_first": share_before_first}]).to_csv(OUTPUT_DIR / "withdrawn_first_due.csv", index=False)

def build_region_odds_ratios():
    df = pd.read_csv(CLEAN_DATA_PATH).dropna(subset=["region", "imd_band", "highest_education", "age_band", "gender", "disability", "num_of_prev_attempts", "studied_credits"])
    df["not_complete"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    
    regions = df["region"].unique()
    baseline_region = "South Region"
    
    results = []
    for r in regions:
        if r == baseline_region:
            results.append({"region": r, "unadjusted_or": 1.0, "adjusted_or": 1.0})
            continue
            
        df_unadj = df[df["region"].isin([baseline_region, r])].copy()
        if len(df_unadj["not_complete"].unique()) > 1:
            X_unadj = (df_unadj["region"] == r).astype(int).values.reshape(-1, 1)
            y_unadj = df_unadj["not_complete"].values
            clf_unadj = LogisticRegression(penalty=None, fit_intercept=True).fit(X_unadj, y_unadj)
            unadj_or = math.exp(clf_unadj.coef_[0][0])
        else:
            unadj_or = 1.0
            
        confounders = ["imd_band", "highest_education", "age_band", "gender", "disability"]
        numeric = ["num_of_prev_attempts", "studied_credits"]
        
        df_adj = df_unadj.copy()
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        X_cat = encoder.fit_transform(df_adj[confounders])
        X_num = df_adj[numeric].values
        X_reg = (df_adj["region"] == r).astype(int).values.reshape(-1, 1)
        
        X_adj = np.hstack([X_reg, X_cat, X_num])
        y_adj = df_adj["not_complete"].values
        
        if len(np.unique(y_adj)) > 1:
            clf_adj = LogisticRegression(penalty=None, fit_intercept=True, max_iter=1000).fit(X_adj, y_adj)
            adj_or = math.exp(clf_adj.coef_[0][0])
        else:
            adj_or = 1.0
            
        results.append({
            "region": r,
            "unadjusted_or": unadj_or,
            "adjusted_or": adj_or
        })
        
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "region_odds_ratios.csv", index=False)

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_no_submission_stats()
    build_unregistration_timeline()
    build_region_odds_ratios()
    print("Page 1 stats computed using clean_dataset.")
