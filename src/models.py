"""
Bree Machine Learning Assessment: Model Training & Selection Module.
Compares the Rule-Based Baseline, Regularized Logistic Regression, and Shallow XGBoost
on a strictly frozen 75/25 stratified holdout.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import build_preprocessor, prepare_dataset


def train_and_evaluate_models(
    df: pd.DataFrame, test_size: float = 0.25, random_seed: int = 42, include_employment: bool = True
) -> Dict[str, Any]:
    """Train candidate models and return predictions on a frozen test holdout."""
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

    # 3. Model 1: Regularized Logistic Regression Pipeline
    preprocessor = build_preprocessor(include_employment=include_employment)
    lr_cv = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 10),
        cv=5,
        penalty="l2",
        scoring="roc_auc",
        random_state=random_seed,
        max_iter=1000,
    )
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", lr_cv),
    ])
    lr_pipeline.fit(X_train, y_train)

    # 4. Model 2: Shallow XGBoost Classifier Pipeline
    # Using preprocessed features (scaled + one-hot encoded) for comparability
    preprocessor_xgb = build_preprocessor(include_employment=include_employment)
    preprocessor_xgb.fit(X_train)
    X_train_trans = preprocessor_xgb.transform(X_train)
    X_test_trans = preprocessor_xgb.transform(X_test)

    # Modest tree structure to prevent overfitting small sample
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

    # 5. Out-of-fold / Test Probabilities
    lr_test_probs = lr_pipeline.predict_proba(X_test)[:, 1]
    xgb_test_probs = xgb_clf.predict_proba(X_test_trans)[:, 1]

    # Baseline: Continuous risk ranker = -rule_score
    rule_test_risk = -rule_test.values

    # Baseline Decisions:
    # Strict policy: denied = default (1), approved + review = repaid (0)
    rule_strict_pred = (raw_test_df["rule_based_decision"] == "denied").astype(int).values
    # Conservative policy: denied + review = default (1), approved = repaid (0)
    rule_cons_pred = (raw_test_df["rule_based_decision"].isin(["denied", "flagged_for_review"])).astype(int).values

    # Determine default decision threshold on training set
    # Using threshold that matches the rule baseline's denial rate or standard 0.5 / F1 optimal
    lr_train_probs = lr_pipeline.predict_proba(X_train)[:, 1]
    xgb_train_probs = xgb_clf.predict_proba(X_train_trans)[:, 1]

    # Threshold matching training default base rate (~0.30)
    thresh_base = float(np.percentile(xgb_train_probs, (1 - y_train.mean()) * 100))

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "raw_test_df": raw_test_df,
        "preprocessor": preprocessor,
        "lr_pipeline": lr_pipeline,
        "xgb_clf": xgb_clf,
        "preprocessor_xgb": preprocessor_xgb,
        "X_test_trans": X_test_trans,
        "rule_test_scores": rule_test.values,
        "rule_test_risk": rule_test_risk,
        "rule_strict_pred": rule_strict_pred,
        "rule_cons_pred": rule_cons_pred,
        "lr_test_probs": lr_test_probs,
        "xgb_test_probs": xgb_test_probs,
        "operating_threshold": thresh_base,
    }


if __name__ == "__main__":
    df_raw = pd.read_csv("data/loan_applications.csv")
    artifacts = train_and_evaluate_models(df_raw)
    print("Models trained successfully.")
    print(f"Test set size: {len(artifacts['y_test'])} (defaults: {artifacts['y_test'].sum()})")
    print(f"Logistic Regression optimal C: {artifacts['lr_pipeline'].named_steps['classifier'].C_[0]:.4f}")
    print(f"Baseline rule test risk shape: {artifacts['rule_test_risk'].shape}")
