"""
Bree Machine Learning Assessment: Employment Group Fairness Audit.

Design: This module consumes the shared artifacts dict from models.py.
It does NOT retrain the primary XGBoost model.  All fairness metrics are
computed from the same y_test, xgb_pred, and operating_threshold used in
evaluate.py, guaranteeing that performance and fairness tables describe
identical predictions.

The employment-ablation model IS retrained here (with explicitly identical
hyperparameters and split) because it requires removing a feature from the
input matrix — but its ROC-AUC, prediction disagreement counts, and
probability differences are all reported so the comparison is explicit.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.features import build_preprocessor, engineer_features, prepare_dataset
from src.models import OPERATING_THRESHOLD


def audit_group_fairness(
    y_true: np.ndarray, y_pred_deny: np.ndarray, group_labels: pd.Series
) -> pd.DataFrame:
    """Compute fairness metrics per employment group."""
    df_eval = pd.DataFrame({
        "y_true":    y_true,
        "y_deny":    y_pred_deny,
        "y_approve": 1 - y_pred_deny,
        "group":     group_labels.values,
    })

    rows = []
    for g, grp in df_eval.groupby("group"):
        n            = len(grp)
        n_defaults   = (grp["y_true"] == 1).sum()
        n_repaid     = (grp["y_true"] == 0).sum()
        approval_rate   = grp["y_approve"].mean()
        default_rate    = grp["y_true"].mean()
        good_denied     = ((grp["y_true"] == 0) & (grp["y_deny"] == 1)).sum()
        good_denial_rate  = good_denied / n_repaid     if n_repaid     > 0 else 0.0
        defaults_caught   = ((grp["y_true"] == 1) & (grp["y_deny"] == 1)).sum()
        default_catch_rate = defaults_caught / n_defaults if n_defaults > 0 else 0.0

        rows.append({
            "group":                     g,
            "count":                     n,
            "actual_default_rate":       round(float(default_rate), 4),
            "approval_rate":             round(float(approval_rate), 4),
            "good_applicant_denial_rate": round(float(good_denial_rate), 4),
            "default_catch_rate":        round(float(default_catch_rate), 4),
            "TP":  int(defaults_caught),
            "FP":  int(good_denied),
            "TN":  int((grp["y_true"] == 0).sum() - good_denied),
            "FN":  int(n_defaults - defaults_caught),
        })
    return pd.DataFrame(rows)


def run_fairness_analysis(
    artifacts: Dict[str, Any],
    df_raw: pd.DataFrame,
    plot_dir: str = "plots",
) -> Dict[str, Any]:
    """Audit fairness from SHARED predictions.  Blinded-model ablation uses
    identical hyperparameters and the same frozen split indices."""
    os.makedirs(plot_dir, exist_ok=True)

    # ── Pull shared objects from artifacts ────────────────────────────────────
    y_test      = artifacts["y_test"].values
    test_groups = artifacts["test_groups"]       # aligned with y_test
    thresh      = artifacts["operating_threshold"]  # == OPERATING_THRESHOLD

    rule_strict_deny = artifacts["rule_strict_pred"]
    rule_cons_deny   = artifacts["rule_cons_pred"]
    xgb_pred_full    = artifacts["xgb_pred"]         # primary model decisions
    xgb_probs_full   = artifacts["xgb_test_probs"]   # primary model probabilities

    # ── Audit primary model (same predictions as evaluate.py) ─────────────────
    audit_rule_strict = audit_group_fairness(y_test, rule_strict_deny, test_groups)
    audit_rule_cons   = audit_group_fairness(y_test, rule_cons_deny,   test_groups)
    audit_xgb_full    = audit_group_fairness(y_test, xgb_pred_full,    test_groups)

    # ── Reconciliation assertions ─────────────────────────────────────────────
    # Sum of per-group TP / FP must equal overall counts.
    assert audit_xgb_full["TP"].sum() == int(xgb_pred_full[y_test == 1].sum()), \
        "Group TP sum does not equal overall TP — predictions are misaligned."
    assert audit_xgb_full["FP"].sum() == int(xgb_pred_full[y_test == 0].sum()), \
        "Group FP sum does not equal overall FP — predictions are misaligned."

    # ── Blinded-model ablation ────────────────────────────────────────────────
    # Retrain with employment_status removed from features.
    # Use same frozen split indices and identical hyperparameters.
    resolved_df = df_raw[df_raw["actual_outcome"] != "ongoing"].copy()
    y_full      = (resolved_df["actual_outcome"] == "defaulted").astype(int)

    idx_train = artifacts["idx_train"]
    idx_test  = artifacts["idx_test"]

    X_blind     = engineer_features(resolved_df, include_employment=False)
    prep_blind  = build_preprocessor(include_employment=False)
    X_train_b   = prep_blind.fit_transform(X_blind.iloc[idx_train])
    X_test_b    = prep_blind.transform   (X_blind.iloc[idx_test])

    xgb_blind = XGBClassifier(
        n_estimators=60, max_depth=3, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        random_state=42, eval_metric="logloss",
    )
    xgb_blind.fit(X_train_b, y_full.iloc[idx_train])
    probs_blind = xgb_blind.predict_proba(X_test_b)[:, 1]
    pred_blind  = (probs_blind >= thresh).astype(int)

    audit_xgb_blind = audit_group_fairness(y_test, pred_blind, test_groups)

    # ── Explicit ablation comparison ──────────────────────────────────────────
    n_disagree       = int(np.count_nonzero(xgb_pred_full != pred_blind))
    max_prob_diff    = float(np.max(np.abs(xgb_probs_full - probs_blind)))
    auc_full         = float(roc_auc_score(y_test, xgb_probs_full))
    auc_blind        = float(roc_auc_score(y_test, probs_blind))

    ablation_comparison = {
        "n_predictions_disagree":   n_disagree,
        "max_abs_prob_difference":  round(max_prob_diff, 4),
        "roc_auc_full_model":       round(auc_full, 4),
        "roc_auc_blinded_model":    round(auc_blind, 4),
    }

    # ── Approval ratios (self-employed / employed) ────────────────────────────
    def se_emp_ratio(audit_df):
        emp  = audit_df.loc[audit_df["group"] == "employed",       "approval_rate"].values
        self = audit_df.loc[audit_df["group"] == "self_employed",  "approval_rate"].values
        if len(emp) == 0 or emp[0] == 0:
            return float("nan")
        return float(self[0] / emp[0])

    # Compute approval rates directly from exact integer counts in audit tables.
    def row_approval_rate(audit_df, group):
        r = audit_df[audit_df["group"] == group]
        if r.empty:
            return float("nan")
        return float(r["approval_rate"].values[0])

    approval_ratios = {
        "Deny-Only (Rule)":              se_emp_ratio(audit_rule_strict),
        "Review-or-Deny (Rule)":         se_emp_ratio(audit_rule_cons),
        "XGBoost (Full Model)":          se_emp_ratio(audit_xgb_full),
        "XGBoost (Blinded: No Employ.)": se_emp_ratio(audit_xgb_blind),
    }

    # Per-group approval rates for the Review-or-Deny policy (for README)
    rev_cons_rates = {
        g: row_approval_rate(audit_rule_cons, g)
        for g in ["employed", "self_employed", "unemployed"]
    }

    # ── Fairness Plot ─────────────────────────────────────────────────────────
    plot_data = []
    for pname, aud in [
        ("Rule Strict",  audit_rule_strict),
        ("Rule Cons",    audit_rule_cons),
        ("XGB Full",     audit_xgb_full),
        ("XGB Blinded",  audit_xgb_blind),
    ]:
        for _, r in aud.iterrows():
            plot_data.append({
                "Policy": pname,
                "Group":  r["group"],
                "Approval Rate":      r["approval_rate"],
                "Good Denied Rate":   r["good_applicant_denial_rate"],
            })
    df_plot = pd.DataFrame(plot_data)

    palette = {"employed": "#2b5c8f", "self_employed": "#d95f02", "unemployed": "#7570b3"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=df_plot, x="Policy", y="Approval Rate",
                hue="Group", palette=palette, ax=axes[0])
    axes[0].set_title("Approval Rate by Employment Group Across Policies",
                       fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Approval Rate"); axes[0].set_ylim(0, 1.05)
    # Note: 0.80 line is the 4/5ths ratio threshold, NOT an absolute rate reference.
    # We omit it from the absolute-rate plot to avoid misinterpretation.

    sns.barplot(data=df_plot, x="Policy", y="Good Denied Rate",
                hue="Group", palette=palette, ax=axes[1])
    axes[1].set_title("Good-Applicant Denial Rate (FPR) Across Policies",
                       fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Rate of Good Applicants Wrongly Denied")
    axes[1].set_ylim(0, 0.7)

    plt.tight_layout()
    fairness_plot_path = os.path.join(plot_dir, "fairness_disparate_impact.png")
    plt.savefig(fairness_plot_path, dpi=300); plt.close()

    return {
        "audit_rule_strict":  audit_rule_strict,
        "audit_rule_cons":    audit_rule_cons,
        "audit_xgb_full":     audit_xgb_full,
        "audit_xgb_blind":    audit_xgb_blind,
        "approval_ratios":    approval_ratios,
        "rev_cons_rates":     rev_cons_rates,
        "ablation_comparison": ablation_comparison,
        "plot": fairness_plot_path,
    }


if __name__ == "__main__":
    from src.models import train_and_evaluate_models
    df_raw   = pd.read_csv("data/loan_applications.csv")
    artifacts = train_and_evaluate_models(df_raw)
    fair_res  = run_fairness_analysis(artifacts, df_raw)

    print("\n--- Employment Group Fairness Audit (Test Set) ---")
    print("\n1. Rule Baseline (Deny-Only):")
    print(fair_res["audit_rule_strict"][
        ["group","count","actual_default_rate","approval_rate",
         "good_applicant_denial_rate","default_catch_rate"]].to_string(index=False))

    print("\n2. XGBoost (Full Model — same predictions as evaluate.py):")
    print(fair_res["audit_xgb_full"][
        ["group","count","actual_default_rate","approval_rate",
         "good_applicant_denial_rate","default_catch_rate"]].to_string(index=False))

    print("\n3. XGBoost (Blinded: No Employment Status):")
    print(fair_res["audit_xgb_blind"][
        ["group","count","actual_default_rate","approval_rate",
         "good_applicant_denial_rate","default_catch_rate"]].to_string(index=False))

    print("\n--- Self-Employed / Employed Approval-Rate Ratios ---")
    for k, v in fair_res["approval_ratios"].items():
        print(f"  {k}: {v:.3f}  (4/5ths reference: 0.80)")

    print("\n--- Ablation Comparison ---")
    ab = fair_res["ablation_comparison"]
    print(f"  Prediction disagreements: {ab['n_predictions_disagree']}")
    print(f"  Max |prob difference|:    {ab['max_abs_prob_difference']:.4f}")
    print(f"  ROC-AUC (full model):     {ab['roc_auc_full_model']:.4f}")
    print(f"  ROC-AUC (blinded model):  {ab['roc_auc_blinded_model']:.4f}")
