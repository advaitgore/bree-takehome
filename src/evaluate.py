"""
Bree Machine Learning Assessment: Evaluation & Business Tradeoff Module.
Computes ROC-AUC, PR-AUC, Precision, Recall, F1, FPR, FNR, confusion matrices with raw counts,
and formalizes the economic cost tradeoff including full reference-policy comparison table.

Assumptions for illustrative contribution calculation (explicitly stated, not validated):
  M = $100   — illustrative contribution margin per approved good loan (fees/interest net of servicing)
  L = $1,000 — illustrative net loss per approved defaulted loan (principal charge-off)
  All figures are scenario calculations, not actual lender profit estimates.
  Recoveries, funding costs, review costs, and operational constraints are omitted.
  All rejected good applicants are treated as permanently lost (no off-take modeled).
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None) -> Dict[str, Any]:
    """Compute precision, recall, F1, FPR, FNR, and confusion matrix counts for default class."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    # FPR = Good applicants wrongly denied / Total good applicants
    total_good = tn + fp
    fpr = fp / total_good if total_good > 0 else 0.0

    # FNR = Defaults that slipped through / Total defaults
    total_defaults = tp + fn
    fnr = fn / total_defaults if total_defaults > 0 else 0.0

    res = {
        "TP (Defaults Caught)": int(tp),
        "FP (Good Wrongly Denied)": int(fp),
        "TN (Good Approved)": int(tn),
        "FN (Defaults Slipped)": int(fn),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "Total Test Count": int(len(y_true)),
    }

    if y_score is not None:
        res["ROC-AUC"] = float(roc_auc_score(y_true, y_score))
        res["PR-AUC"] = float(average_precision_score(y_true, y_score))

    return res


def evaluate_all(artifacts: Dict[str, Any], plot_dir: str = "plots") -> Dict[str, Any]:
    """Evaluate Rule Baseline, Logistic Regression, and XGBoost on the frozen test set."""
    os.makedirs(plot_dir, exist_ok=True)
    y_test = artifacts["y_test"].values

    # Continuous risk scores
    rule_risk = artifacts["rule_test_risk"]  # -rule_score
    lr_probs = artifacts["lr_test_probs"]
    xgb_probs = artifacts["xgb_test_probs"]

    # Binary decisions
    rule_strict = artifacts["rule_strict_pred"]
    rule_cons = artifacts["rule_cons_pred"]

    # Operating threshold for ML models (matched to train base rate or optimal threshold)
    thresh = artifacts.get("operating_threshold", 0.35)
    lr_pred = (lr_probs >= thresh).astype(int)
    xgb_pred = (xgb_probs >= thresh).astype(int)

    # Compute metrics for each policy
    metrics = {
        "Rule Baseline (Strict: Deny Only)": compute_binary_metrics(y_test, rule_strict, rule_risk),
        "Rule Baseline (Conservative: Review+Deny)": compute_binary_metrics(y_test, rule_cons, rule_risk),
        "Logistic Regression (thresh=0.35)": compute_binary_metrics(y_test, lr_pred, lr_probs),
        "XGBoost Classifier (thresh=0.35)": compute_binary_metrics(y_test, xgb_pred, xgb_probs),
    }

    # 1. Plot ROC and PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC Curves
    for name, score, color in [
        ("Rule Baseline (-Score)", rule_risk, "#7f7f7f"),
        ("Logistic Regression", lr_probs, "#1f77b4"),
        ("XGBoost", xgb_probs, "#2ca02c"),
    ]:
        fpr_arr, tpr_arr, _ = roc_curve(y_test, score)
        auc_val = roc_auc_score(y_test, score)
        axes[0].plot(fpr_arr, tpr_arr, label=f"{name} (AUC = {auc_val:.3f})", color=color, lw=2)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
    axes[0].set_title("ROC Curves (Default Discrimination)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (Good Wrongly Denied)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Defaults Caught)", fontsize=11)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curves
    baseline_prev = y_test.mean()
    for name, score, color in [
        ("Rule Baseline (-Score)", rule_risk, "#7f7f7f"),
        ("Logistic Regression", lr_probs, "#1f77b4"),
        ("XGBoost", xgb_probs, "#2ca02c"),
    ]:
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, score)
        pr_auc = average_precision_score(y_test, score)
        axes[1].plot(rec_arr, prec_arr, label=f"{name} (PR-AUC = {pr_auc:.3f})", color=color, lw=2)

    axes[1].axhline(baseline_prev, color="k", linestyle="--", alpha=0.5, label=f"Prevalence ({baseline_prev:.2f})")
    axes[1].set_title("Precision-Recall Curves (Default Class)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall (Defaults Caught)", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    roc_pr_path = os.path.join(plot_dir, "roc_pr_curves.png")
    plt.savefig(roc_pr_path, dpi=300)
    plt.close()

    # 2. Plot Confusion Matrices
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    configs = [
        ("Rule (Strict)", rule_strict, axes[0]),
        ("Rule (Conservative)", rule_cons, axes[1]),
        ("Logistic Regression", lr_pred, axes[2]),
        ("XGBoost", xgb_pred, axes[3]),
    ]

    for title, preds, ax in configs:
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Repaid (0)", "Default (1)"],
            yticklabels=["Repaid (0)", "Default (1)"],
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Action", fontsize=10)
        ax.set_ylabel("Actual Outcome", fontsize=10)

    plt.tight_layout()
    cm_path = os.path.join(plot_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 3. Illustrative Contribution Table (full reference-policy comparison)
    # -------------------------------------------------------------------------
    # Assumptions (illustrative scenario, not actual lender profit):
    #   M = $100  per approved good loan (fees/interest net of servicing)
    #   L = $1000 per approved defaulted loan (principal charge-off)
    #   Approval of a rejected good applicant is assumed lost (zero off-take).
    #   Recoveries, review costs, funding costs, and operational constraints omitted.
    # -------------------------------------------------------------------------
    M = 100   # illustrative good-loan margin
    L = 1000  # illustrative default loss

    n_defaults_total = int(y_test.sum())             # 136
    n_good_total     = int((1 - y_test).sum())       # 323

    def contribution(tn, fn):
        """Approved good loans × M minus approved defaults × L."""
        good_approved    = tn
        defaults_approved = fn
        return good_approved * M - defaults_approved * L

    # Reference policies
    approve_all_contrib  = contribution(n_good_total, n_defaults_total)  # TN=323, FN=136
    reject_all_contrib   = 0  # no loans approved

    rows_econ = []

    # Approve-all and reject-all as bookends
    rows_econ.append({
        "Policy": "Approve Everyone (reference)",
        "Good Approved (TN)": n_good_total,
        "Defaults Approved (FN)": n_defaults_total,
        "Illustrative Contribution": approve_all_contrib,
    })

    for name, m in metrics.items():
        tn = m["TN (Good Approved)"]
        fn = m["FN (Defaults Slipped)"]
        rows_econ.append({
            "Policy": name,
            "Good Approved (TN)": tn,
            "Defaults Approved (FN)": fn,
            "Illustrative Contribution": contribution(tn, fn),
        })

    rows_econ.append({
        "Policy": "Reject Everyone (reference)",
        "Good Approved (TN)": 0,
        "Defaults Approved (FN)": 0,
        "Illustrative Contribution": reject_all_contrib,
    })

    contribution_table = pd.DataFrame(rows_econ)

    # 4. Incremental analysis: XGBoost vs. Deny-Only Baseline
    base_tp = metrics["Rule Baseline (Strict: Deny Only)"]["TP (Defaults Caught)"]
    base_fp = metrics["Rule Baseline (Strict: Deny Only)"]["FP (Good Wrongly Denied)"]
    xgb_tp  = metrics["XGBoost Classifier (thresh=0.35)"]["TP (Defaults Caught)"]
    xgb_fp  = metrics["XGBoost Classifier (thresh=0.35)"]["FP (Good Wrongly Denied)"]

    delta_tp = xgb_tp - base_tp
    delta_fp = xgb_fp - base_fp
    break_even_ratio = delta_fp / delta_tp if delta_tp > 0 else 0.0

    # 5. Break-even L/M for Review-or-Deny vs. XGBoost (critique item 1)
    rev_tp = metrics["Rule Baseline (Conservative: Review+Deny)"]["TP (Defaults Caught)"]
    rev_fp = metrics["Rule Baseline (Conservative: Review+Deny)"]["FP (Good Wrongly Denied)"]
    rev_vs_xgb_delta_tp = rev_tp - xgb_tp   # +25
    rev_vs_xgb_delta_fp = rev_fp - xgb_fp   # +90
    rev_vs_xgb_break_even = (
        rev_vs_xgb_delta_fp / rev_vs_xgb_delta_tp
        if rev_vs_xgb_delta_tp > 0 else float("inf")
    )

    economics = {
        "M_assumption": M,
        "L_assumption": L,
        "contribution_table": contribution_table,
        "xgb_vs_deny_only": {
            "base_defaults_caught": base_tp,
            "model_defaults_caught": xgb_tp,
            "delta_defaults_caught": delta_tp,
            "base_good_denied": base_fp,
            "model_good_denied": xgb_fp,
            "delta_good_denied": delta_fp,
            "break_even_loss_margin_ratio": float(break_even_ratio),
        },
        "review_deny_vs_xgb": {
            "delta_defaults_caught": rev_vs_xgb_delta_tp,
            "delta_good_denied": rev_vs_xgb_delta_fp,
            "break_even_loss_margin_ratio": float(rev_vs_xgb_break_even),
            "note": (
                "Review-or-Deny catches more defaults than XGBoost at 0.35 "
                f"(+{rev_vs_xgb_delta_tp} TP) but denies more good applicants "
                f"(+{rev_vs_xgb_delta_fp} FP). Under assumed L/M={L//M}x, "
                "Review-or-Deny shows higher illustrative contribution. "
                "XGBoost at 0.35 was selected as an F1-balanced operating point, "
                "not the economically dominant one under these assumptions."
            ),
        },
    }

    return {
        "metrics": metrics,
        "economics": economics,
        "plots": [roc_pr_path, cm_path],
    }


if __name__ == "__main__":
    from src.models import train_and_evaluate_models
    df_raw = pd.read_csv("data/loan_applications.csv")
    artifacts = train_and_evaluate_models(df_raw)
    eval_res = evaluate_all(artifacts)

    print("\n--- Model Performance Comparison (Frozen Test Set N=459) ---")
    summary_rows = []
    for k, v in eval_res["metrics"].items():
        summary_rows.append({
            "Policy / Model": k,
            "ROC-AUC": f"{v.get('ROC-AUC', 0.0):.3f}" if "ROC-AUC" in v else "N/A",
            "PR-AUC": f"{v.get('PR-AUC', 0.0):.3f}" if "PR-AUC" in v else "N/A",
            "Precision": f"{v['Precision']:.3f}",
            "Recall": f"{v['Recall']:.3f}",
            "F1": f"{v['F1']:.3f}",
            "FPR (Good Denied)": f"{v['FPR']:.3f}",
            "FNR (Defaults Slipped)": f"{v['FNR']:.3f}",
            "Defaults Caught (TP)": v["TP (Defaults Caught)"],
            "Good Denied (FP)": v["FP (Good Wrongly Denied)"],
        })
    print(pd.DataFrame(summary_rows).to_string(index=False))

    econ = eval_res["economics"]
    print(f"\n--- Illustrative Contribution Table (M=${econ['M_assumption']}, L=${econ['L_assumption']}) ---")
    print("(Scenario calculations using holdout counts — not actual lender profit)")
    tbl = econ["contribution_table"].copy()
    tbl["Illustrative Contribution"] = tbl["Illustrative Contribution"].apply(lambda x: f"${x:,.0f}")
    print(tbl.to_string(index=False))

    print("\n--- Incremental Analysis: XGBoost vs. Deny-Only ---")
    xd = econ["xgb_vs_deny_only"]
    print(f"  Delta Defaults Caught: +{xd['delta_defaults_caught']}")
    print(f"  Delta Good Denied:     +{xd['delta_good_denied']}")
    print(f"  Break-even L/M for XGBoost vs. Deny-Only: {xd['break_even_loss_margin_ratio']:.2f}x")

    print("\n--- Review-or-Deny vs. XGBoost (Critique Note) ---")
    rd = econ["review_deny_vs_xgb"]
    print(f"  Review-or-Deny catches +{rd['delta_defaults_caught']} more defaults than XGBoost")
    print(f"  but denies +{rd['delta_good_denied']} more good applicants.")
    print(f"  Break-even L/M for Review-or-Deny superiority: {rd['break_even_loss_margin_ratio']:.2f}x")
    print(f"  Note: {rd['note']}")
