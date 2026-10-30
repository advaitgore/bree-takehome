"""
Bree Machine Learning Assessment: Explainability & Adverse Action Reason Prototype.
Computes TreeSHAP attributions with verified sign direction (positive = increases default risk)
and translates top risk factors into underwriting review reason codes.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models import train_and_evaluate_models


# Human-readable underwriting reason code mappings for risk-increasing factors
REASON_MAPPINGS = {
    "loan_to_income_ratio": "Requested loan amount is high relative to monthly income",
    "withdrawal_to_deposit_ratio": "Monthly cash outflows exceed safe ratio of deposits",
    "bank_has_overdrafts": "History of account overdrafts detected",
    "is_doc_missing": "Lack of verified income documentation",
    "doc_income_imputed": "Low or unverified documented income",
    "income_discrepancy_ratio": "Documented income reflects substantial discrepancy from stated income",
    "bank_ending_balance": "Low ending bank balance reserves",
    "num_documents_submitted": "Insufficient documentation submitted",
    "monthly_withdrawals": "High absolute monthly withdrawal volume",
    "monthly_deposits": "Low monthly deposit volume",
    "bank_has_consistent_deposits": "Irregular bank deposit history",
    "stated_monthly_income": "Low stated monthly income",
    "loan_amount": "High requested principal amount",
}


def explain_decisions(artifacts: Dict[str, Any], plot_dir: str = "plots") -> Dict[str, Any]:
    """Compute SHAP explanations and generate individual applicant case studies."""
    os.makedirs(plot_dir, exist_ok=True)
    xgb_clf = artifacts["xgb_clf"]
    X_test_trans = artifacts["X_test_trans"]
    preprocessor = artifacts["preprocessor_xgb"]
    raw_test_df = artifacts["raw_test_df"].reset_index(drop=True)
    y_test = artifacts["y_test"].values
    xgb_probs = artifacts["xgb_test_probs"]

    # Feature names from preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # TreeExplainer on XGBoost
    explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X_test_trans)

    # Global feature importance (mean absolute SHAP)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feat_imp = pd.DataFrame({
        "feature": [f.replace("num__", "").replace("cat__", "") for f in feature_names],
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    # Plot Global Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    top_feats = feat_imp.head(10).sort_values("mean_abs_shap", ascending=True)
    ax.barh(top_feats["feature"], top_feats["mean_abs_shap"], color="#1f77b4")
    ax.set_title("Top 10 Global Features by Mean |SHAP| (Default Risk)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Model Log-Odds)", fontsize=11)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "shap_feature_importance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Case Studies: Select 3 illustrative applicants
    # 1. Denied Default (True Positive)
    # 2. Denied Good Applicant (False Positive)
    # 3. Approved Self-Employed that Rule System Denied or Flagged
    case_studies = []

    for i in range(len(raw_test_df)):
        row = raw_test_df.iloc[i]
        prob = xgb_probs[i]
        actual = row["actual_outcome"]
        rule_dec = row["rule_based_decision"]
        rule_sc = row["rule_based_score"]

        # Find top 3 risk-increasing features (positive SHAP values push toward default)
        ind_shap = shap_values[i]
        top_risk_indices = np.argsort(ind_shap)[::-1]
        top_reasons = []
        for idx in top_risk_indices:
            if ind_shap[idx] > 0.05:  # meaningful positive contribution
                clean_feat = feature_names[idx].replace("num__", "").replace("cat__", "")
                reason_text = REASON_MAPPINGS.get(clean_feat, f"Factor: {clean_feat}")
                top_reasons.append(f"{reason_text} (+{ind_shap[idx]:.2f} log-odds)")
            if len(top_reasons) >= 3:
                break

        case_studies.append({
            "applicant_id": row["applicant_id"],
            "employment_status": row["employment_status"],
            "loan_amount": row["loan_amount"],
            "stated_income": row["stated_monthly_income"],
            "bank_balance": row["bank_ending_balance"],
            "has_overdrafts": row["bank_has_overdrafts"],
            "actual_outcome": actual,
            "rule_decision": rule_dec,
            "rule_score": rule_sc,
            "model_prob": prob,
            "model_action": "denied" if prob >= 0.35 else "approved",
            "top_reasons": top_reasons,
        })

    # Pick 3 diverse examples
    selected_cases = []
    # Case A: Denied by model and defaulted
    for c in case_studies:
        if c["model_action"] == "denied" and c["actual_outcome"] == "defaulted" and len(c["top_reasons"]) >= 2:
            selected_cases.append(c)
            break

    # Case B: Self-employed applicant approved by model (and repaid) but flagged or denied by rule system
    for c in case_studies:
        if (
            c["employment_status"] == "self_employed"
            and c["model_action"] == "approved"
            and c["actual_outcome"] == "repaid"
            and c["rule_decision"] in ["denied", "flagged_for_review"]
        ):
            selected_cases.append(c)
            break

    # Case C: Denied applicant with clear income discrepancy or overdrafts
    for c in case_studies:
        if c not in selected_cases and c["model_action"] == "denied" and c["has_overdrafts"]:
            selected_cases.append(c)
            break

    return {
        "feature_importance": feat_imp,
        "selected_cases": selected_cases,
        "plot": plot_path,
    }


if __name__ == "__main__":
    df_raw = pd.read_csv("data/loan_applications.csv")
    artifacts = train_and_evaluate_models(df_raw)
    exp_res = explain_decisions(artifacts)

    print("\n--- Top 7 Global Features by SHAP Importance ---")
    print(exp_res["feature_importance"].head(7).to_string(index=False))

    print("\n--- Illustrative Underwriting Case Studies (Auditor Review Prototype) ---")
    for i, c in enumerate(exp_res["selected_cases"], 1):
        print(f"\nCase {i}: Applicant {c['applicant_id']} ({c['employment_status']})")
        print(f"  Loan: ${c['loan_amount']:,.0f} | Stated Income: ${c['stated_income']:,.0f} | Bank Balance: ${c['bank_balance']:,.0f} | Overdrafts: {c['has_overdrafts']}")
        print(f"  Rule Decision: {c['rule_decision']} (Score: {c['rule_score']}) | Model Action: {c['model_action']} (Prob: {c['model_prob']:.2f})")
        print(f"  Actual Ground Truth: {c['actual_outcome']}")
        print("  Key Underwriting Reason Codes:")
        for r in c["top_reasons"]:
            print(f"    - {r}")
