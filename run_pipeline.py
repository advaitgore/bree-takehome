"""
Bree Machine Learning Assessment: Master Pipeline CLI Runner.
Executes data generation, EDA, model training, evaluation, fairness audits,
and explainability case studies in a single reproducible command.

One model is trained once.  evaluate.py, fairness.py, and explainability.py
all consume the same artifacts dict, guaranteeing consistent predictions and
threshold across every reported table.
"""
import json
import os
import sys
import time
import pandas as pd

from src.generate_data import generate_loan_data
from src.eda import run_eda
from src.models import train_and_evaluate_models, OPERATING_THRESHOLD
from src.evaluate import evaluate_all
from src.fairness import run_fairness_analysis
from src.explainability import explain_decisions


def main():
    start_time = time.time()
    print("=" * 80)
    print(" BREE ML TAKE-HOME: LOAN UNDERWRITING PIPELINE")
    print("=" * 80)

    # 1. Dataset
    csv_path = "data/loan_applications.csv"
    if not os.path.exists(csv_path):
        print("\n[Step 1/6] Generating synthetic dataset (seed 42)...")
        generate_loan_data(n=2000, seed=42, output_path=csv_path)
    else:
        print(f"\n[Step 1/6] Found existing dataset at {csv_path}.")
    df_raw = pd.read_csv(csv_path)
    print(f"Loaded {len(df_raw)} applications. "
          f"Outcomes: {df_raw['actual_outcome'].value_counts().to_dict()}")

    # 2. EDA
    print("\n[Step 2/6] Exploratory Data Analysis & Generator Audit...")
    run_eda(csv_path=csv_path, plot_dir="plots")

    # 3. Train — ONE model, ONE threshold, returned in artifacts
    print(f"\n[Step 3/6] Training models (operating threshold fixed at "
          f"{OPERATING_THRESHOLD})...")
    artifacts = train_and_evaluate_models(df_raw, test_size=0.25, random_seed=42)
    n_test    = len(artifacts["y_test"])
    n_def     = int(artifacts["y_test"].sum())
    n_rep     = n_test - n_def
    print(f"  Test set: N={n_test} ({n_def} defaults, {n_rep} repaid)")
    print(f"  LR best C: {artifacts['lr_best_C']:.4f}")

    # 4. Evaluate — consumes artifacts, no retraining
    print("\n[Step 4/6] Evaluating test performance...")
    eval_res = evaluate_all(artifacts, plot_dir="plots")

    print("\n" + "-" * 80)
    print(f"TEST SET PERFORMANCE (N={n_test}, threshold={artifacts['operating_threshold']})")
    print("-" * 80)
    rows = []
    for name, m in eval_res["metrics"].items():
        rows.append({
            "Policy / Model":       name,
            "ROC-AUC":              f"{m['ROC-AUC']:.3f}" if "ROC-AUC" in m else "—",
            "PR-AUC":               f"{m['PR-AUC']:.3f}"  if "PR-AUC"  in m else "—",
            "Precision":            f"{m['Precision']:.3f}",
            "Recall":               f"{m['Recall']:.3f}",
            "F1":                   f"{m['F1']:.3f}",
            "FPR":                  f"{m['FPR']:.3f}",
            "FNR":                  f"{m['FNR']:.3f}",
            "TP":                   m["TP (Defaults Caught)"],
            "FP":                   m["FP (Good Wrongly Denied)"],
            "TN":                   m["TN (Good Approved)"],
            "FN":                   m["FN (Defaults Slipped)"],
        })
    print(pd.DataFrame(rows).to_string(index=False))

    econ = eval_res["economics"]
    xd   = econ["xgb_vs_deny_only"]
    rd   = econ["review_deny_vs_xgb"]
    print("\n" + "-" * 80)
    print(f"ILLUSTRATIVE CONTRIBUTION (M=${econ['M_assumption']}, "
          f"L=${econ['L_assumption']} — scenario, not actual profit)")
    print("-" * 80)
    tbl = econ["contribution_table"].copy()
    tbl["Illustrative Contribution ($)"] = (
        tbl["Illustrative Contribution ($)"].apply(lambda x: f"${x:,.0f}"))
    print(tbl.to_string(index=False))
    print(f"\n  XGBoost vs. Deny-Only:       Delta_TP=+{xd['delta_defaults_caught']}, "
          f"Delta_FP=+{xd['delta_good_denied']}, break-even L/M={xd['break_even_lm_ratio']:.2f}x")
    print(f"  Review-or-Deny vs. XGBoost:  Delta_TP=+{rd['delta_defaults_caught']}, "
          f"Delta_FP=+{rd['delta_good_denied']}, break-even L/M={rd['break_even_lm_ratio']:.2f}x")
    print(f"  [XGBoost threshold={artifacts['operating_threshold']:.2f} selected as "
          f"balanced operating point, not EV-optimised]")

    # 5. Fairness — consumes same artifacts (no retraining of primary model)
    print("\n[Step 5/6] Employment Group Fairness Audit & Ablation...")
    fair_res = run_fairness_analysis(artifacts, df_raw, plot_dir="plots")

    print("\n  Self-Employed / Employed Approval-Rate Ratios (4/5ths ref = 0.80):")
    for k, v in fair_res["approval_ratios"].items():
        flag = "[PASS]" if v >= 0.80 else "[FAIL]"
        print(f"  {flag} {k}: {v:.3f}")

    ab = fair_res["ablation_comparison"]
    print(f"\n  Ablation (employment feature removed):")
    print(f"    Prediction disagreements:  {ab['n_predictions_disagree']}")
    print(f"    Max |prob difference|:     {ab['max_abs_prob_difference']:.4f}")
    print(f"    ROC-AUC full:   {ab['roc_auc_full_model']:.4f}")
    print(f"    ROC-AUC blind:  {ab['roc_auc_blinded_model']:.4f}")

    # 6. Explainability
    print("\n[Step 6/6] SHAP attributions & case studies...")
    exp_res = explain_decisions(artifacts, plot_dir="plots")
    print(f"  SHAP base value (log-odds): {exp_res['base_value']:.4f}")
    print(f"  Additivity MAE: {exp_res['additivity_mae']:.6f}")
    print("\n  Case Studies (outcome-selected illustrations):")
    for i, c in enumerate(exp_res["selected_cases"], 1):
        print(f"\n  Case {i} [{c.get('case_label', '')}]")
        print(f"    {c['applicant_id']} ({c['employment_status']}) | "
              f"Loan ${c['loan_amount']:,.0f} | Balance ${c['bank_balance']:,.0f} | "
              f"Overdrafts: {c['has_overdrafts']}")
        print(f"    Rule: {c['rule_decision']} (score {c['rule_score']:.1f}) | "
              f"Model: {c['model_action']} (prob {c['model_prob']:.2f}) | "
              f"Outcome: {c['actual_outcome']}")
        for r in c["top_reasons"]:
            print(f"    - {r}")

    # 7. Export machine-readable metrics
    os.makedirs("output", exist_ok=True)
    metrics_export = {}
    for k, v in eval_res["metrics"].items():
        metrics_export[k] = {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}
    metrics_export["_meta"] = {
        "operating_threshold": float(artifacts["operating_threshold"]),
        "test_n": n_test, "test_defaults": n_def, "test_repaid": n_rep,
    }
    with open("output/metrics.json", "w") as f:
        json.dump(metrics_export, f, indent=2)

    fair_export = {}
    for policy, aud_df in [
        ("deny_only_rule",     fair_res["audit_rule_strict"]),
        ("review_deny_rule",   fair_res["audit_rule_cons"]),
        ("xgb_full",           fair_res["audit_xgb_full"]),
        ("xgb_blinded",        fair_res["audit_xgb_blind"]),
    ]:
        fair_export[policy] = aud_df.to_dict(orient="records")
    fair_export["approval_ratios"]     = fair_res["approval_ratios"]
    fair_export["ablation_comparison"] = fair_res["ablation_comparison"]
    with open("output/fairness.json", "w") as f:
        json.dump(fair_export, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" PIPELINE COMPLETE IN {elapsed:.2f}s")
    print(f" Plots: plots/   Machine-readable outputs: output/")
    print("=" * 80)


if __name__ == "__main__":
    main()
