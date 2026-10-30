"""
Bree Machine Learning Assessment: Explainability & Adverse Action Reason Prototype.

Uses TreeSHAP (log-odds space) from the single fitted XGBoost in artifacts.
Positive SHAP values increase predicted default log-odds relative to the base value.

Reason codes are NEUTRAL descriptions of model contributions — they do NOT
assert that a specific threshold was breached or that an observable rule was
violated.  They are illustrative reviewer reason codes, not legally compliant
adverse-action notices.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models import OPERATING_THRESHOLD


# Neutral reason-code phrases: describe the model's SHAP contribution, not a
# factual assertion about whether a threshold was breached.
REASON_MAPPINGS = {
    "loan_to_income_ratio":           "Loan-to-income ratio contributed positively to estimated risk",
    "withdrawal_to_deposit_ratio":    "Withdrawal-to-deposit ratio contributed positively to estimated risk",
    "bank_has_overdrafts":            "Overdraft indicator contributed positively to estimated risk",
    "is_doc_missing":                 "Absence of income documentation contributed positively to estimated risk",
    "doc_income_imputed":             "Documented income level contributed positively to estimated risk",
    "income_discrepancy_ratio":       "Stated-to-documented income ratio contributed positively to estimated risk",
    "bank_ending_balance":            "Bank ending balance contributed positively to estimated risk",
    "num_documents_submitted":        "Number of documents submitted contributed positively to estimated risk",
    "monthly_withdrawals":            "Monthly withdrawal volume contributed positively to estimated risk",
    "monthly_deposits":               "Monthly deposit volume contributed positively to estimated risk",
    "bank_has_consistent_deposits":   "Deposit consistency indicator contributed positively to estimated risk",
    "stated_monthly_income":          "Stated monthly income contributed positively to estimated risk",
    "loan_amount":                    "Requested loan amount contributed positively to estimated risk",
}


def explain_decisions(artifacts: Dict[str, Any], plot_dir: str = "plots") -> Dict[str, Any]:
    """Compute SHAP explanations using the shared fitted XGBoost model.

    All decisions use artifacts['operating_threshold'] — never a hardcoded value.
    """
    os.makedirs(plot_dir, exist_ok=True)

    xgb_clf      = artifacts["xgb_clf"]
    X_test_trans = artifacts["X_test_trans"]
    preprocessor = artifacts["preprocessor_xgb"]
    raw_test_df  = artifacts["raw_test_df"].reset_index(drop=True)
    y_test       = artifacts["y_test"].values
    xgb_probs    = artifacts["xgb_test_probs"]
    xgb_pred     = artifacts["xgb_pred"]          # pre-computed at shared threshold
    thresh       = artifacts["operating_threshold"]

    feature_names = preprocessor.get_feature_names_out()

    # ── TreeSHAP ──────────────────────────────────────────────────────────────
    explainer   = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X_test_trans)
    base_value  = float(explainer.expected_value)

    # ── SHAP additivity assertion (log-odds space) ────────────────────────────
    # For each observation, base_value + sum(shap_values[i]) should equal the
    # model's raw log-odds output.  We verify the mean absolute error is tiny.
    import warnings
    logit_preds = np.log(xgb_probs / (1 - np.clip(xgb_probs, 1e-7, 1 - 1e-7)))
    shap_sums   = shap_values.sum(axis=1) + base_value
    additivity_mae = float(np.mean(np.abs(shap_sums - logit_preds)))
    assert additivity_mae < 0.01, (
        f"SHAP additivity violated: mean |error| = {additivity_mae:.4f} — "
        "check TreeExplainer model_output setting."
    )

    # ── Global Feature Importance ─────────────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feat_imp = pd.DataFrame({
        "feature": [f.replace("num__", "").replace("cat__", "") for f in feature_names],
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_feats = feat_imp.head(10).sort_values("mean_abs_shap", ascending=True)
    ax.barh(top_feats["feature"], top_feats["mean_abs_shap"], color="#1f77b4")
    ax.set_title("Top 10 Features by Mean |SHAP| (log-odds contribution to default risk)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Absolute SHAP Value (log-odds)")
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "shap_feature_importance.png")
    plt.savefig(plot_path, dpi=300); plt.close()

    # ── Build per-applicant case records ──────────────────────────────────────
    case_studies = []
    for i in range(len(raw_test_df)):
        row      = raw_test_df.iloc[i]
        prob     = float(xgb_probs[i])
        actual   = row["actual_outcome"]
        rule_dec = row["rule_based_decision"]
        rule_sc  = row["rule_based_score"]
        decision = "denied" if xgb_pred[i] == 1 else "approved"

        ind_shap = shap_values[i]
        top_risk_indices = np.argsort(ind_shap)[::-1]
        top_reasons: List[str] = []
        for idx in top_risk_indices:
            if ind_shap[idx] > 0.05:
                clean_feat  = feature_names[idx].replace("num__", "").replace("cat__", "")
                reason_text = REASON_MAPPINGS.get(clean_feat, f"Feature '{clean_feat}' contributed positively")
                top_reasons.append(f"{reason_text} (+{ind_shap[idx]:.2f} log-odds)")
            if len(top_reasons) >= 3:
                break

        case_studies.append({
            "idx":               i,
            "applicant_id":      row["applicant_id"],
            "employment_status": row["employment_status"],
            "loan_amount":       row["loan_amount"],
            "stated_income":     row["stated_monthly_income"],
            "bank_balance":      row["bank_ending_balance"],
            "has_overdrafts":    row["bank_has_overdrafts"],
            "actual_outcome":    actual,
            "rule_decision":     rule_dec,
            "rule_score":        rule_sc,
            "model_prob":        prob,
            "model_action":      decision,
            "top_reasons":       top_reasons,
        })

    # ── Select 3 outcome-labelled illustrations ───────────────────────────────
    # Cases are deliberately chosen for auditor review — they are NOT
    # representative samples; they are illustrative of specific error modes.
    selected_cases = []

    # A. True Positive (correctly denied defaulter)
    for c in case_studies:
        if c["model_action"] == "denied" and c["actual_outcome"] == "defaulted" and len(c["top_reasons"]) >= 2:
            c["case_label"] = "True Positive (denied, defaulted) — outcome-selected illustration"
            selected_cases.append(c); break

    # B. False Positive (good applicant wrongly denied — illustrates cost of over-rejection)
    for c in case_studies:
        if c["model_action"] == "denied" and c["actual_outcome"] == "repaid" and len(c["top_reasons"]) >= 1:
            c["case_label"] = "False Positive (denied, repaid) — outcome-selected illustration"
            selected_cases.append(c); break

    # C. Self-employed applicant ML approved but rule flagged/denied — illustrates fairness recovery
    for c in case_studies:
        if (c not in selected_cases
                and c["employment_status"] == "self_employed"
                and c["model_action"] == "approved"
                and c["actual_outcome"] == "repaid"
                and c["rule_decision"] in ["denied", "flagged_for_review"]):
            c["case_label"] = ("Self-employed approved by ML, flagged by rule — "
                               "outcome-selected illustration")
            selected_cases.append(c); break

    return {
        "feature_importance":  feat_imp,
        "selected_cases":      selected_cases,
        "base_value":          base_value,
        "additivity_mae":      additivity_mae,
        "operating_threshold": thresh,
        "plot": plot_path,
    }


if __name__ == "__main__":
    from src.models import train_and_evaluate_models
    df_raw    = pd.read_csv("data/loan_applications.csv")
    artifacts = train_and_evaluate_models(df_raw)
    exp_res   = explain_decisions(artifacts)

    print(f"\nSHAP base value (log-odds): {exp_res['base_value']:.4f}")
    print(f"Additivity MAE: {exp_res['additivity_mae']:.6f} (should be near 0)")
    print(f"Operating threshold: {exp_res['operating_threshold']}")

    print("\n--- Top 7 Features by Mean |SHAP| ---")
    print(exp_res["feature_importance"].head(7).to_string(index=False))

    print("\n--- Illustrative Case Studies (outcome-selected; not representative) ---")
    for i, c in enumerate(exp_res["selected_cases"], 1):
        print(f"\nCase {i} [{c['case_label']}]")
        print(f"  Applicant: {c['applicant_id']} ({c['employment_status']})")
        print(f"  Loan ${c['loan_amount']:,.0f} | Income ${c['stated_income']:,.0f} "
              f"| Balance ${c['bank_balance']:,.0f} | Overdrafts: {c['has_overdrafts']}")
        print(f"  Rule: {c['rule_decision']} (score {c['rule_score']:.1f}) | "
              f"Model: {c['model_action']} (prob {c['model_prob']:.2f})")
        print(f"  Outcome: {c['actual_outcome']}")
        print("  Top risk contributors (SHAP, log-odds):")
        for r in c["top_reasons"]:
            print(f"    - {r}")
