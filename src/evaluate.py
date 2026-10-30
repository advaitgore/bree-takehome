"""
Bree Machine Learning Assessment: Evaluation & Business Tradeoff Module.
Computes ROC-AUC, PR-AUC, Precision, Recall, F1, FPR, FNR, confusion matrices,
and an illustrative full-reference-policy contribution table.

All binary decisions are produced at the single OPERATING_THRESHOLD defined in
src/models.py and passed through artifacts.  This module never redefines the
threshold.

Assumptions for illustrative contribution calculation (explicitly stated):
  M = $100   — illustrative contribution margin per approved good loan
  L = $1,000 — illustrative net loss per approved defaulted loan (principal charge-off)
  Recoveries, review costs, funding costs, and operational constraints omitted.
  All rejected good applicants are treated as permanently lost (no off-take modelled).
  These are scenario calculations using holdout counts — not actual lender profit.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any
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

from src.models import OPERATING_THRESHOLD


def compute_binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None
) -> Dict[str, Any]:
    """Compute precision, recall, F1, FPR, FNR, and confusion matrix counts."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall    = recall_score   (y_true, y_pred, pos_label=1, zero_division=0)
    f1        = f1_score       (y_true, y_pred, pos_label=1, zero_division=0)

    total_good     = tn + fp
    total_defaults = tp + fn
    fpr = fp / total_good     if total_good     > 0 else 0.0
    fnr = fn / total_defaults if total_defaults > 0 else 0.0

    res = {
        "TP (Defaults Caught)":    int(tp),
        "FP (Good Wrongly Denied)": int(fp),
        "TN (Good Approved)":      int(tn),
        "FN (Defaults Slipped)":   int(fn),
        "Precision": float(precision),
        "Recall":    float(recall),
        "F1":        float(f1),
        "FPR":       float(fpr),
        "FNR":       float(fnr),
        "Total Test Count": int(len(y_true)),
    }
    if y_score is not None:
        res["ROC-AUC"] = float(roc_auc_score(y_true, y_score))
        res["PR-AUC"]  = float(average_precision_score(y_true, y_score))
    return res


def evaluate_all(artifacts: Dict[str, Any], plot_dir: str = "plots") -> Dict[str, Any]:
    """Evaluate all policies on the frozen test set using shared predictions."""
    os.makedirs(plot_dir, exist_ok=True)

    y_test    = artifacts["y_test"].values
    thresh    = artifacts["operating_threshold"]   # single shared value

    # Continuous scores
    rule_risk  = artifacts["rule_test_risk"]
    lr_probs   = artifacts["lr_test_probs"]
    xgb_probs  = artifacts["xgb_test_probs"]

    # Binary decisions — pre-computed in models.py at the shared threshold
    rule_strict = artifacts["rule_strict_pred"]
    rule_cons   = artifacts["rule_cons_pred"]
    lr_pred     = artifacts["lr_pred"]
    xgb_pred    = artifacts["xgb_pred"]

    # Labels derived from the actual threshold (not hardcoded strings)
    thresh_label = f"{thresh:.2f}"
    lr_label  = f"Logistic Regression (thresh={thresh_label})"
    xgb_label = f"XGBoost (thresh={thresh_label})"

    metrics = {
        "Deny-Only (Rule)":        compute_binary_metrics(y_test, rule_strict, rule_risk),
        "Review-or-Deny (Rule)":   compute_binary_metrics(y_test, rule_cons,   rule_risk),
        lr_label:                  compute_binary_metrics(y_test, lr_pred,      lr_probs),
        xgb_label:                 compute_binary_metrics(y_test, xgb_pred,     xgb_probs),
    }

    # ── Reconciliation assertions ─────────────────────────────────────────────
    # Fairness module uses same xgb_pred; assert totals will match.
    assert metrics[xgb_label]["TP (Defaults Caught)"] == int(xgb_pred[y_test == 1].sum()), \
        "XGBoost TP mismatch — check that predictions are shared correctly."
    assert metrics[xgb_label]["FP (Good Wrongly Denied)"] == int(xgb_pred[y_test == 0].sum()), \
        "XGBoost FP mismatch."

    # ── ROC and PR Curves ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, score, color in [
        ("Rule Baseline (-Score)",  rule_risk,  "#7f7f7f"),
        ("Logistic Regression",     lr_probs,   "#1f77b4"),
        ("XGBoost",                 xgb_probs,  "#2ca02c"),
    ]:
        fpr_arr, tpr_arr, _ = roc_curve(y_test, score)
        auc_val = roc_auc_score(y_test, score)
        axes[0].plot(fpr_arr, tpr_arr,
                     label=f"{name} (AUC={auc_val:.3f})", color=color, lw=2)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
    axes[0].set_title("ROC Curves (Default Discrimination)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (Good Wrongly Denied)")
    axes[0].set_ylabel("True Positive Rate (Defaults Caught)")
    axes[0].legend(loc="lower right"); axes[0].grid(True, alpha=0.3)

    baseline_prev = y_test.mean()
    for name, score, color in [
        ("Rule Baseline (-Score)",  rule_risk,  "#7f7f7f"),
        ("Logistic Regression",     lr_probs,   "#1f77b4"),
        ("XGBoost",                 xgb_probs,  "#2ca02c"),
    ]:
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, score)
        pr_auc = average_precision_score(y_test, score)
        axes[1].plot(rec_arr, prec_arr,
                     label=f"{name} (PR-AUC={pr_auc:.3f})", color=color, lw=2)
    axes[1].axhline(baseline_prev, color="k", linestyle="--", alpha=0.5,
                    label=f"Prevalence ({baseline_prev:.2f})")
    axes[1].set_title("Precision-Recall Curves (Default Class)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall (Defaults Caught)")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right"); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    roc_pr_path = os.path.join(plot_dir, "roc_pr_curves.png")
    plt.savefig(roc_pr_path, dpi=300); plt.close()

    # ── Confusion Matrices ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for (title, preds), ax in zip([
        ("Rule (Deny-Only)",      rule_strict),
        ("Rule (Review+Deny)",    rule_cons),
        ("Logistic Regression",   lr_pred),
        ("XGBoost",               xgb_pred),
    ], axes):
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Repaid", "Default"],
                    yticklabels=["Repaid", "Default"])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(plot_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300); plt.close()

    # ── Illustrative Contribution Table ──────────────────────────────────────
    M = 100    # illustrative good-loan margin ($)
    L = 1000   # illustrative default loss ($)
    n_defaults_total = int(y_test.sum())
    n_good_total     = int((1 - y_test).sum())

    def contribution(tn, fn):
        return tn * M - fn * L

    rows_econ = [{"Policy": "Approve Everyone (reference)",
                  "Good Approved (TN)": n_good_total,
                  "Defaults Approved (FN)": n_defaults_total,
                  "Illustrative Contribution ($)": contribution(n_good_total, n_defaults_total)}]

    for name, m in metrics.items():
        rows_econ.append({
            "Policy": name,
            "Good Approved (TN)": m["TN (Good Approved)"],
            "Defaults Approved (FN)": m["FN (Defaults Slipped)"],
            "Illustrative Contribution ($)": contribution(
                m["TN (Good Approved)"], m["FN (Defaults Slipped)"]),
        })

    rows_econ.append({"Policy": "Reject Everyone (reference)",
                      "Good Approved (TN)": 0,
                      "Defaults Approved (FN)": 0,
                      "Illustrative Contribution ($)": 0})

    contribution_table = pd.DataFrame(rows_econ)

    # Incremental comparisons
    base_m  = metrics["Deny-Only (Rule)"]
    xgb_m   = metrics[xgb_label]
    rev_m   = metrics["Review-or-Deny (Rule)"]

    d_tp_xgb_vs_deny = xgb_m["TP (Defaults Caught)"]   - base_m["TP (Defaults Caught)"]
    d_fp_xgb_vs_deny = xgb_m["FP (Good Wrongly Denied)"] - base_m["FP (Good Wrongly Denied)"]
    be_xgb_vs_deny   = d_fp_xgb_vs_deny / d_tp_xgb_vs_deny if d_tp_xgb_vs_deny > 0 else float("inf")

    d_tp_rev_vs_xgb = rev_m["TP (Defaults Caught)"]    - xgb_m["TP (Defaults Caught)"]
    d_fp_rev_vs_xgb = rev_m["FP (Good Wrongly Denied)"] - xgb_m["FP (Good Wrongly Denied)"]
    be_rev_vs_xgb   = d_fp_rev_vs_xgb / d_tp_rev_vs_xgb if d_tp_rev_vs_xgb > 0 else float("inf")

    economics = {
        "M_assumption": M,
        "L_assumption": L,
        "contribution_table": contribution_table,
        "xgb_label": xgb_label,
        "xgb_vs_deny_only": {
            "delta_defaults_caught": d_tp_xgb_vs_deny,
            "delta_good_denied":     d_fp_xgb_vs_deny,
            "break_even_lm_ratio":   float(be_xgb_vs_deny),
        },
        "review_deny_vs_xgb": {
            "delta_defaults_caught": d_tp_rev_vs_xgb,
            "delta_good_denied":     d_fp_rev_vs_xgb,
            "break_even_lm_ratio":   float(be_rev_vs_xgb),
        },
    }

    return {"metrics": metrics, "economics": economics, "plots": [roc_pr_path, cm_path]}


if __name__ == "__main__":
    from src.models import train_and_evaluate_models
    df_raw = pd.read_csv("data/loan_applications.csv")
    artifacts = train_and_evaluate_models(df_raw)
    eval_res = evaluate_all(artifacts)

    print(f"\n--- Model Performance (Test N={len(artifacts['y_test'])}) ---")
    rows = []
    for k, v in eval_res["metrics"].items():
        rows.append({
            "Policy / Model":       k,
            "ROC-AUC":              f"{v.get('ROC-AUC', 0):.3f}" if "ROC-AUC" in v else "—",
            "PR-AUC":               f"{v.get('PR-AUC',  0):.3f}" if "PR-AUC"  in v else "—",
            "Precision":            f"{v['Precision']:.3f}",
            "Recall":               f"{v['Recall']:.3f}",
            "F1":                   f"{v['F1']:.3f}",
            "FPR":                  f"{v['FPR']:.3f}",
            "FNR":                  f"{v['FNR']:.3f}",
            "Defaults Caught (TP)": v["TP (Defaults Caught)"],
            "Good Denied (FP)":     v["FP (Good Wrongly Denied)"],
        })
    print(pd.DataFrame(rows).to_string(index=False))

    econ = eval_res["economics"]
    print(f"\n--- Illustrative Contribution (M=${econ['M_assumption']}, "
          f"L=${econ['L_assumption']}) — scenario only ---")
    tbl = econ["contribution_table"].copy()
    tbl["Illustrative Contribution ($)"] = (
        tbl["Illustrative Contribution ($)"].apply(lambda x: f"${x:,.0f}"))
    print(tbl.to_string(index=False))

    xd = econ["xgb_vs_deny_only"]
    rd = econ["review_deny_vs_xgb"]
    print(f"\nXGBoost vs. Deny-Only: ΔTP=+{xd['delta_defaults_caught']}, "
          f"ΔFP=+{xd['delta_good_denied']}, break-even L/M={xd['break_even_lm_ratio']:.2f}x")
    print(f"Review-or-Deny vs. XGBoost: ΔTP=+{rd['delta_defaults_caught']}, "
          f"ΔFP=+{rd['delta_good_denied']}, break-even L/M={rd['break_even_lm_ratio']:.2f}x")
