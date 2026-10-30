"""
Bree Machine Learning Assessment: Feature Engineering & Preprocessing Module.
Enforces strict predictor allowlisting, leakage guards, and safe mathematical transforms.
"""
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Strict Allowlist of Predictor Features (observable at application time)
PREDICTOR_ALLOWLIST: List[str] = [
    "stated_monthly_income",
    "loan_amount",
    "bank_ending_balance",
    "bank_has_overdrafts",
    "bank_has_consistent_deposits",
    "monthly_withdrawals",
    "monthly_deposits",
    "num_documents_submitted",
    "is_doc_missing",
    "doc_income_imputed",
    "income_discrepancy_ratio",
    "withdrawal_to_deposit_ratio",
    "loan_to_income_ratio",
    "employment_status",
]

# Strict Denylist to prevent data leakage
STRICT_LEAKAGE_DENYLIST: List[str] = [
    "applicant_id",
    "actual_outcome",
    "days_to_default",
    "rule_based_score",
    "rule_based_decision",
]


def engineer_features(df: pd.DataFrame, include_employment: bool = True) -> pd.DataFrame:
    """Engineer observable underwriting features from raw application data.
    
    Includes boundary guards on denominators and handles missing documentation.
    """
    X = pd.DataFrame(index=df.index)

    # Raw Financial Features
    X["stated_monthly_income"] = df["stated_monthly_income"].astype(float)
    X["loan_amount"] = df["loan_amount"].astype(float)
    X["bank_ending_balance"] = df["bank_ending_balance"].astype(float)
    X["bank_has_overdrafts"] = df["bank_has_overdrafts"].astype(int)
    X["bank_has_consistent_deposits"] = df["bank_has_consistent_deposits"].astype(int)
    X["monthly_withdrawals"] = df["monthly_withdrawals"].astype(float)
    X["monthly_deposits"] = df["monthly_deposits"].astype(float)
    X["num_documents_submitted"] = df["num_documents_submitted"].astype(float)

    # Missing Documentation Indicators & Imputation
    doc_missing = df["documented_monthly_income"].isna()
    X["is_doc_missing"] = doc_missing.astype(int)
    X["doc_income_imputed"] = df["documented_monthly_income"].fillna(0.0).astype(float)

    # Observed Discrepancy Ratio (bounded and neutral when docs are missing)
    has_valid_doc = (~doc_missing) & (df["documented_monthly_income"] > 0)
    X["income_discrepancy_ratio"] = np.where(
        has_valid_doc,
        df["stated_monthly_income"] / np.maximum(df["documented_monthly_income"], 1.0),
        1.0,  # neutral baseline when undocumented
    ).clip(0.1, 20.0)

    # Cashflow & Debt Burden Ratios with Denominator Floors
    X["withdrawal_to_deposit_ratio"] = (
        df["monthly_withdrawals"] / np.maximum(df["monthly_deposits"], 1.0)
    ).clip(0.0, 10.0)

    X["loan_to_income_ratio"] = (
        df["loan_amount"] / np.maximum(df["stated_monthly_income"], 1.0)
    ).clip(0.0, 5.0)

    # Employment Category
    if include_employment:
        X["employment_status"] = df["employment_status"].astype(str)

    # Leakage check: verify no forbidden column entered
    for col in STRICT_LEAKAGE_DENYLIST:
        if col in X.columns:
            raise ValueError(f"CRITICAL: Leakage column '{col}' detected in feature set!")

    return X


def build_preprocessor(include_employment: bool = True) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for scaling and encoding."""
    numeric_features = [
        "stated_monthly_income",
        "loan_amount",
        "bank_ending_balance",
        "bank_has_overdrafts",
        "bank_has_consistent_deposits",
        "monthly_withdrawals",
        "monthly_deposits",
        "num_documents_submitted",
        "is_doc_missing",
        "doc_income_imputed",
        "income_discrepancy_ratio",
        "withdrawal_to_deposit_ratio",
        "loan_to_income_ratio",
    ]

    transformers = [
        ("num", StandardScaler(), numeric_features),
    ]

    if include_employment:
        categorical_features = ["employment_status"]
        transformers.append(
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_dataset(
    df: pd.DataFrame, include_employment: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare features, binary target (1=defaulted, 0=repaid), and rule scores for resolved loans."""
    # Exclude ongoing applications for supervised learning
    resolved_df = df[df["actual_outcome"] != "ongoing"].copy()

    y = (resolved_df["actual_outcome"] == "defaulted").astype(int)
    rule_scores = resolved_df["rule_based_score"].astype(float)
    X = engineer_features(resolved_df, include_employment=include_employment)

    return X, y, rule_scores


if __name__ == "__main__":
    test_df = pd.read_csv("data/loan_applications.csv")
    X, y, rule_scores = prepare_dataset(test_df)
    print("Features engineered successfully.")
    print(f"X shape: {X.shape}, y shape: {y.shape}, default prevalence: {y.mean():.3f}")
    print(f"Feature columns: {list(X.columns)}")
