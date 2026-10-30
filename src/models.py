"""
Bree Machine Learning Assessment: Model Training & Selection Module.
Compares the Rule-Based Baseline, Regularized Logistic Regression, and Shallow XGBoost
on a strictly frozen 75/25 stratified holdout.

Design principles
-----------------
* ONE primary model is fit here.  evaluate.py, fairness.py, and explainability.py all
  consume the returned artifacts — they never retrain the primary model.
* ONE explicit threshold (OPERATING_THRESHOLD = 0.35) is defined here and exported so
  every downstream module uses the same decision boundary.
* Logistic regression cross-validates the full preprocessing + classifier pipeline so
  that scaling statistics are not fitted on validation-fold observations during internal
  C selection.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import build_preprocessor, prepare_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Single, explicitly fixed operating threshold used by ALL downstream modules.
# Selected as a balanced operating point approximating the training default
# prevalence (~30%).  This is NOT cost-optimised; under a pure EV criterion with
# L=$1,000 and M=$100 the approval-favourable boundary is p < M/(M+L) ≈ 9%.
# Threshold was fixed before test-set inspection.
# ─────────────────────────────────────────────────────────────────────────────
OPERATING_THRESHOLD: float = 0.35


def train_and_evaluate_models(
    df: pd.DataFrame, test_size: float = 0.25, random_seed: int = 42,
    include_employment: bool = True,
) -> Dict[str, Any]:
    """Train candidate models and return predictions on a frozen test holdout.

    Returns a single artifacts dict consumed by evaluate.py, fairness.py, and
    explainability.py.  Only this function fits the primary XGBoost and LR models.
    """
    # 1. Prepare resolved dataset
    resolved_df = df[df["actual_outcome"] != "ongoing"].copy()
    X, y, rule_scores = prepare_dataset(df, include_employment=include_employment)

    # 2. Strict Frozen Train/Test Split (stratified on target)
    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, random_state=random_seed, stratify=y
    )

    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
    rule_train, rule_test = rule_scores.iloc[idx_train], rule_scores.iloc[idx_test]
    raw_test_df = resolved_df.iloc[idx_test].copy()

    # ── Model 1: Logistic Regression ──────────────────────────────────────────
    # Cross-validate the FULL pipeline (preprocessor + classifier) so that
    # scaling is fitted inside each CV fold, not on the full development set.
    # This prevents internal CV leakage of scaling statistics.
    lr_pipeline_cv = Pipeline([
        ("preprocessor", build_preprocessor(include_employment=include_employment)),
        ("classifier", LogisticRegression(penalty="l2", max_iter=1000,
                                          random_state=random_seed)),
    ])
    param_grid = {"classifier__C": np.logspace(-3, 2, 10).tolist()}
    lr_gs = GridSearchCV(
        lr_pipeline_cv, param_grid, cv=5, scoring="roc_auc", refit=True, n_jobs=-1
    )
    lr_gs.fit(X_train, y_train)
    lr_pipeline = lr_gs.best_estimator_
    best_C = lr_gs.best_params_["classifier__C"]

    # ── Model 2: Shallow XGBoost ──────────────────────────────────────────────
    # Fit preprocessor once on training data; transform train and test.
    preprocessor_xgb = build_preprocessor(include_employment=include_employment)
    preprocessor_xgb.fit(X_train)
    X_train_trans = preprocessor_xgb.transform(X_train)
    X_test_trans  = preprocessor_xgb.transform(X_test)

    xgb_clf = XGBClassifier(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        random_state=random_seed,
        eval_metric="logloss",
    )
    xgb_clf.fit(X_train_trans, y_train)

    # ── Test probabilities ────────────────────────────────────────────────────
    lr_test_probs  = lr_pipeline.predict_proba(X_test)[:, 1]
    xgb_test_probs = xgb_clf.predict_proba(X_test_trans)[:, 1]

    # Binary decisions at the single shared threshold
    lr_pred  = (lr_test_probs  >= OPERATING_THRESHOLD).astype(int)
    xgb_pred = (xgb_test_probs >= OPERATING_THRESHOLD).astype(int)

    # Baseline decisions
    rule_test_risk   = -rule_test.values
    rule_strict_pred = (raw_test_df["rule_based_decision"] == "denied").astype(int).values
    rule_cons_pred   = (raw_test_df["rule_based_decision"]
                        .isin(["denied", "flagged_for_review"])).astype(int).values

    # ── Employment group labels for the test set ──────────────────────────────
    test_groups = resolved_df.iloc[idx_test]["employment_status"].reset_index(drop=True)

    return {
        # Raw data
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "raw_test_df": raw_test_df,
        "test_groups": test_groups,
        "idx_train": idx_train,
        "idx_test": idx_test,
        # Models & preprocessors
        "lr_pipeline": lr_pipeline,
        "lr_best_C": best_C,
        "xgb_clf": xgb_clf,
        "preprocessor_xgb": preprocessor_xgb,
        "X_train_trans": X_train_trans,
        "X_test_trans": X_test_trans,
        # Continuous scores
        "rule_test_scores": rule_test.values,
        "rule_test_risk": rule_test_risk,
        "lr_test_probs": lr_test_probs,
        "xgb_test_probs": xgb_test_probs,
        # Binary decisions (all at OPERATING_THRESHOLD)
        "rule_strict_pred": rule_strict_pred,
        "rule_cons_pred": rule_cons_pred,
        "lr_pred": lr_pred,
        "xgb_pred": xgb_pred,
        # Shared threshold
        "operating_threshold": OPERATING_THRESHOLD,
    }


if __name__ == "__main__":
    df_raw = pd.read_csv("data/loan_applications.csv")
    artifacts = train_and_evaluate_models(df_raw)
    print("Models trained successfully.")
    print(f"Test set size: {len(artifacts['y_test'])} "
          f"(defaults: {artifacts['y_test'].sum()})")
    print(f"Logistic Regression best C: {artifacts['lr_best_C']:.4f}")
    print(f"Operating threshold (all models): {artifacts['operating_threshold']}")
