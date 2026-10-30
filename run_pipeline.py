"""
Bree Machine Learning Assessment: Master Pipeline CLI Runner.
Executes data generation, EDA, model training, evaluation, fairness audits,
and explainability case studies in a single reproducible command.
"""
import os
import sys
import time
import pandas as pd

from src.generate_data import generate_loan_data
from src.eda import run_eda
from src.models import train_and_evaluate_models
from src.evaluate import evaluate_all
from src.fairness import run_fairness_analysis
from src.explainability import explain_decisions


def main():
    start_time = time.time()
    print("=" * 80)
    print(" BREE ML TAKE-HOME: AI-POWERED LOAN APPLICATION PROCESSOR")
    print("=" * 80)

    # 1. Dataset Generation
    csv_path = "data/loan_applications.csv"
    if not os.path.exists(csv_path):
        print("\n[Step 1/5] Generating synthetic dataset with seed 42...")
        generate_loan_data(n=2000, seed=42, output_path=csv_path)
    else:
        print(f"\n[Step 1/5] Found existing dataset at {csv_path}.")

    df_raw = pd.read_csv(csv_path)
    print(f"Loaded {len(df_raw)} applications. Outcomes: {df_raw['actual_outcome'].value_counts().to_dict()}")

    # 2. Exploratory Data Analysis
    print("\n[Step 2/5] Running Exploratory Data Analysis & Generator Audit...")
    eda_res = run_eda(csv_path=csv_path, plot_dir="plots")

    # 3. Model Training & Selection
    print("\n[Step 3/5] Training candidate models on frozen 75/25 holdout...")
    artifacts = train_and_evaluate_models(df_raw, test_size=0.25, random_seed=42)

    # 4. Evaluation & Tradeoff Economics
    print("\n[Step 4/5] Evaluating test performance, confusion matrices & business economics...")
    eval_res = evaluate_all(artifacts, plot_dir="plots")

    print("\n" + "-" * 80)
    print("TEST SET PERFORMANCE COMPARISON (N=459 Resolved Applications)")
    print("-" * 80)
    rows = []
    for name, m in eval_res["metrics"].items():
        rows.append({
            "Policy / Model": name,
            "ROC-AUC": f"{m.get('ROC-AUC', 0.0):.3f}" if "ROC-AUC" in m else "N/A",
            "PR-AUC": f"{m.get('PR-AUC', 0.0):.3f}" if "PR-AUC" in m else "N/A",
            "Precision": f"{m['Precision']:.3f}",
            "Recall": f"{m['Recall']:.3f}",
            "F1": f"{m['F1']:.3f}",
            "FPR (Good Denied)": f"{m['FPR']:.3f}",
            "FNR (Defaults Slipped)": f"{m['FNR']:.3f}",
            "Defaults Caught (TP)": m["TP (Defaults Caught)"],
            "Good Denied (FP)": m["FP (Good Wrongly Denied)"],
        })
    metrics_table = pd.DataFrame(rows)
    print(metrics_table.to_string(index=False))

    econ = eval_res["economics"]
    xd = econ["xgb_vs_deny_only"]
    rd = econ["review_deny_vs_xgb"]
    print("\n" + "-" * 80)
    print(f"ILLUSTRATIVE CONTRIBUTION TABLE (M=${econ['M_assumption']}, L=${econ['L_assumption']} per loan — scenario, not actual profit)")
    print("-" * 80)
    tbl = econ["contribution_table"].copy()
    tbl["Illustrative Contribution"] = tbl["Illustrative Contribution"].apply(lambda x: f"${x:,.0f}")
    print(tbl.to_string(index=False))
    print(f"\n  XGBoost vs. Deny-Only: +{xd['delta_defaults_caught']} defaults caught, +{xd['delta_good_denied']} good denied, break-even L/M = {xd['break_even_loss_margin_ratio']:.2f}x")
    print(f"  Review-or-Deny vs. XGBoost: +{rd['delta_defaults_caught']} defaults, +{rd['delta_good_denied']} good denied, break-even L/M = {rd['break_even_loss_margin_ratio']:.2f}x")
    print(f"  Note: XGBoost (0.35) selected for F1-balanced performance, not economic dominance. See README §3.")

    # 5. Fairness Audit
    print("\n[Step 5/5] Conducting Employment Group Fairness Audit & Ablation...")
    fair_res = run_fairness_analysis(df_raw, plot_dir="plots")
    print("\nApproval Rate Ratios (Self-Employed vs. Employed):")
    for k, v in fair_res["approval_ratios"].items():
        print(f"  • {k:40s}: {v:.3f} (Benchmark: 0.80)")

    # 6. Explainability Case Studies
    print("\n[Step 6/6] Generating SHAP attributions & Adverse Action reason case studies...")
    exp_res = explain_decisions(artifacts, plot_dir="plots")
    print("\nRepresentative Underwriting Review Cases:")
    for i, c in enumerate(exp_res["selected_cases"], 1):
        print(f"\nCase {i} ({c['applicant_id']} - {c['employment_status']}):")
        print(f"  • Loan: ${c['loan_amount']:,.0f} | Stated Income: ${c['stated_income']:,.0f} | Balance: ${c['bank_balance']:,.0f} | Overdrafts: {c['has_overdrafts']}")
        print(f"  • Rule Decision: {c['rule_decision']} (Score: {c['rule_score']:.1f}) | Model Action: {c['model_action']} (Prob: {c['model_prob']:.2f})")
        print(f"  • Actual Outcome: {c['actual_outcome']}")
        print(f"  • Primary Adverse Action Reasons: {'; '.join(c['top_reasons'])}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print(f" Diagnostic plots saved in: plots/")
    print("=" * 80)


if __name__ == "__main__":
    main()
