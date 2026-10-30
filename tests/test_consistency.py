"""
Targeted consistency tests for the Bree take-home pipeline.
Verifies that performance, fairness, and explainability evaluate the exact same
predictions and that per-group totals reconcile to overall totals.
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import train_and_evaluate_models, OPERATING_THRESHOLD
from src.evaluate import evaluate_all
from src.fairness import run_fairness_analysis
from src.explainability import explain_decisions


@pytest.fixture(scope="module")
def pipeline_artifacts():
    data_path = "data/loan_applications.csv"
    if not os.path.exists(data_path):
        from src.generate_data import generate_loan_data
        generate_loan_data(n=2000, seed=42, output_path=data_path)
    df_raw = pd.read_csv(data_path)
    artifacts = train_and_evaluate_models(df_raw, test_size=0.25, random_seed=42)
    eval_res = evaluate_all(artifacts)
    fair_res = run_fairness_analysis(artifacts, df_raw)
    exp_res = explain_decisions(artifacts)
    return {
        "df_raw": df_raw,
        "artifacts": artifacts,
        "eval_res": eval_res,
        "fair_res": fair_res,
        "exp_res": exp_res,
    }


def test_shared_operating_threshold(pipeline_artifacts):
    """Verify that evaluate, fairness, and explainability use the single fixed threshold."""
    art = pipeline_artifacts["artifacts"]
    exp = pipeline_artifacts["exp_res"]
    assert art["operating_threshold"] == OPERATING_THRESHOLD
    assert exp["operating_threshold"] == OPERATING_THRESHOLD


def test_confusion_matrix_reconciliation(pipeline_artifacts):
    """Verify sum of per-group TP and FP equals overall performance table TP and FP."""
    fair_df = pipeline_artifacts["fair_res"]["audit_xgb_full"]
    eval_metrics = pipeline_artifacts["eval_res"]["metrics"]
    xgb_metrics = eval_metrics[f"XGBoost (thresh={OPERATING_THRESHOLD:.2f})"]

    group_tp_sum = fair_df["TP"].sum()
    group_fp_sum = fair_df["FP"].sum()
    group_tn_sum = fair_df["TN"].sum()
    group_fn_sum = fair_df["FN"].sum()

    assert group_tp_sum == xgb_metrics["TP (Defaults Caught)"]
    assert group_fp_sum == xgb_metrics["FP (Good Wrongly Denied)"]
    assert group_tn_sum == xgb_metrics["TN (Good Approved)"]
    assert group_fn_sum == xgb_metrics["FN (Defaults Slipped)"]
    assert group_tp_sum + group_fp_sum + group_tn_sum + group_fn_sum == len(pipeline_artifacts["artifacts"]["y_test"])


def test_shap_additivity(pipeline_artifacts):
    """Verify TreeSHAP additivity in log-odds space."""
    assert pipeline_artifacts["exp_res"]["additivity_mae"] < 0.01


def test_no_leakage_in_features(pipeline_artifacts):
    """Verify no target or leakage columns exist in feature matrix."""
    forbidden = ["actual_outcome", "days_to_default", "applicant_id", "rule_based_score", "rule_based_decision"]
    for col in forbidden:
        assert col not in pipeline_artifacts["artifacts"]["X_train"].columns
        assert col not in pipeline_artifacts["artifacts"]["X_test"].columns
