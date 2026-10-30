"""
Bree Machine Learning Assessment: Employment Group Fairness Audit.
Audits disparate impact across employment groups, compares full vs. blinded models,
and disentangles the direct 6-point rule penalty from indirect proxy correlations.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier

from src.features import build_preprocessor, engineer_features, prepare_dataset


def audit_group_fairness(
    y_true: np.ndarray, y_pred_deny: np.ndarray, group_labels: pd.Series
) -> pd.DataFrame:
    """Compute plain-language fairness metrics per group."""
    df_eval = pd.DataFrame({
        "y_true": y_true,
        "y_deny": y_pred_deny,
        "y_approve": 1 - y_pred_deny,
        "group": group_labels.values,
    })

    rows = []
    for g, grp in df_eval.groupby("group"):
        n = len(grp)
        n_defaults = (grp["y_true"] == 1).sum()
        n_repaid = (grp["y_true"] == 0).sum()

        approval_rate = grp["y_approve"].mean()
        default_rate = grp["y_true"].mean()

        # Good-applicant denial rate (FPR within group)
        good_denied = ((grp["y_true"] == 0) & (grp["y_deny"] == 1)).sum()
        good_denial_rate = good_denied / n_repaid if n_repaid > 0 else 0.0

        # Default catch rate (Recall within group)
        defaults_caught = ((grp["y_true"] == 1) & (grp["y_deny"] == 1)).sum()
        default_catch_rate = defaults_caught / n_defaults if n_defaults > 0 else 0.0

        rows.append({
            "group": g,
            "count": n,
            "actual_default_rate": default_rate,
            "approval_rate": approval_rate,
            "good_applicant_denial_rate": good_denial_rate,
            "default_catch_rate": default_catch_rate,
        })

    res_df = pd.DataFrame(rows)
    return res_df


def run_fairness_analysis(df_raw: pd.DataFrame, plot_dir: str = "plots") -> Dict[str, Any]:
    """Train full model and blinded model, evaluate fairness across employment groups."""
    os.makedirs(plot_dir, exist_ok=True)
    resolved_df = df_raw[df_raw["actual_outcome"] != "ongoing"].copy()

    # Common frozen split
    y = (resolved_df["actual_outcome"] == "defaulted").astype(int)
    groups = resolved_df["employment_status"]
    from sklearn.model_selection import train_test_split
    idx_train, idx_test = train_test_split(
        np.arange(len(resolved_df)), test_size=0.25, random_state=42, stratify=y
    )

    test_groups = groups.iloc[idx_test]
    y_test = y.iloc[idx_test].values
    raw_test_df = resolved_df.iloc[idx_test]

    # Baseline Decisions
    rule_strict_deny = (raw_test_df["rule_based_decision"] == "denied").astype(int).values
    rule_cons_deny = (raw_test_df["rule_based_decision"].isin(["denied", "flagged_for_review"])).astype(int).values

    # Full Model (with employment status)
    X_full = engineer_features(resolved_df, include_employment=True)
    prep_full = build_preprocessor(include_employment=True)
    X_train_full = prep_full.fit_transform(X_full.iloc[idx_train])
    X_test_full = prep_full.transform(X_full.iloc[idx_test])

    xgb_full = XGBClassifier(
        n_estimators=60, max_depth=3, learning_rate=0.05, subsample=0.85, random_state=42, eval_metric="logloss"
    )
    xgb_full.fit(X_train_full, y.iloc[idx_train])
    p_full = xgb_full.predict_proba(X_test_full)[:, 1]
    xgb_full_deny = (p_full >= 0.35).astype(int)

    # Blinded Model (employment status removed)
    X_blind = engineer_features(resolved_df, include_employment=False)
    prep_blind = build_preprocessor(include_employment=False)
    X_train_blind = prep_blind.fit_transform(X_blind.iloc[idx_train])
    X_test_blind = prep_blind.transform(X_blind.iloc[idx_test])

    xgb_blind = XGBClassifier(
        n_estimators=60, max_depth=3, learning_rate=0.05, subsample=0.85, random_state=42, eval_metric="logloss"
    )
    xgb_blind.fit(X_train_blind, y.iloc[idx_train])
    p_blind = xgb_blind.predict_proba(X_test_blind)[:, 1]
    xgb_blind_deny = (p_blind >= 0.35).astype(int)

    # Compute audits
    audit_rule_strict = audit_group_fairness(y_test, rule_strict_deny, test_groups)
    audit_rule_cons = audit_group_fairness(y_test, rule_cons_deny, test_groups)
    audit_xgb_full = audit_group_fairness(y_test, xgb_full_deny, test_groups)
    audit_xgb_blind = audit_group_fairness(y_test, xgb_blind_deny, test_groups)

    # Calculate 4/5ths approval ratios (self-employed vs employed)
    def calc_ratio(audit_df):
        emp_app = audit_df[audit_df["group"] == "employed"]["approval_rate"].values[0]
        self_app = audit_df[audit_df["group"] == "self_employed"]["approval_rate"].values[0]
        return self_app / emp_app if emp_app > 0 else 0.0

    ratios = {
        "Rule Baseline (Strict)": calc_ratio(audit_rule_strict),
        "Rule Baseline (Conservative)": calc_ratio(audit_rule_cons),
        "XGBoost (Full Model)": calc_ratio(audit_xgb_full),
        "XGBoost (Blinded: No Employment)": calc_ratio(audit_xgb_blind),
    }

    # Plot Fairness Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_data = []
    for policy_name, aud in [
        ("Rule Strict", audit_rule_strict),
        ("Rule Cons", audit_rule_cons),
        ("XGB Full", audit_xgb_full),
        ("XGB Blinded", audit_xgb_blind),
    ]:
        for _, r in aud.iterrows():
            plot_data.append({
                "Policy": policy_name,
                "Group": r["group"],
                "Approval Rate": r["approval_rate"],
                "Good Denied Rate": r["good_applicant_denial_rate"],
            })
    df_plot = pd.DataFrame(plot_data)

    sns.barplot(
        data=df_plot,
        x="Policy",
        y="Approval Rate",
        hue="Group",
        palette={"employed": "#2b5c8f", "self_employed": "#d95f02", "unemployed": "#7570b3"},
        ax=axes[0],
    )
    axes[0].set_title("Approval Rate by Employment Group Across Policies", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Approval Rate", fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(0.8, color="gray", linestyle=":", alpha=0.6, label="80% Level")

    sns.barplot(
        data=df_plot,
        x="Policy",
        y="Good Denied Rate",
        hue="Group",
        palette={"employed": "#2b5c8f", "self_employed": "#d95f02", "unemployed": "#7570b3"},
        ax=axes[1],
    )
    axes[1].set_title("Good-Applicant Denial Rate (FPR) Across Policies", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Rate of Good Applicants Wrongly Denied", fontsize=11)
    axes[1].set_ylim(0, 0.7)

    plt.tight_layout()
    fairness_plot_path = os.path.join(plot_dir, "fairness_disparate_impact.png")
    plt.savefig(fairness_plot_path, dpi=300)
    plt.close()

    return {
        "audit_rule_strict": audit_rule_strict,
        "audit_rule_cons": audit_rule_cons,
        "audit_xgb_full": audit_xgb_full,
        "audit_xgb_blind": audit_xgb_blind,
        "approval_ratios": ratios,
        "plot": fairness_plot_path,
    }


if __name__ == "__main__":
    df_raw = pd.read_csv("data/loan_applications.csv")
    fair_res = run_fairness_analysis(df_raw)

    print("\n--- Employment Group Fairness Audit (Test Set N=459) ---")
    print("\n1. Rule Baseline (Strict):")
    print(fair_res["audit_rule_strict"].to_string(index=False))

    print("\n2. XGBoost (Full Model):")
    print(fair_res["audit_xgb_full"].to_string(index=False))

    print("\n3. XGBoost (Blinded: No Employment Status):")
    print(fair_res["audit_xgb_blind"].to_string(index=False))

    print("\n--- Self-Employed to Employed Approval Rate Ratios ---")
    for k, v in fair_res["approval_ratios"].items():
        print(f"  {k}: {v:.3f} (Reference: 0.80)")
