"""
Bree Machine Learning Assessment: Exploratory Data Analysis (EDA) Module.
Audits the synthetic dataset, compares generator code with prompt prose,
examines missingness, discrepancy, ongoing loans, and outputs diagnostic plots.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def run_eda(csv_path: str = "data/loan_applications.csv", plot_dir: str = "plots") -> dict:
    """Run exploratory data analysis and generate diagnostic plots."""
    os.makedirs(plot_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # 1. Basic Counts & Proportions
    total_n = len(df)
    outcome_counts = df["actual_outcome"].value_counts().to_dict()
    outcome_props = (df["actual_outcome"].value_counts(normalize=True) * 100).round(2).to_dict()

    # Resolved vs Ongoing
    resolved_df = df[df["actual_outcome"] != "ongoing"].copy()
    resolved_n = len(resolved_df)
    resolved_default_rate = (resolved_df["actual_outcome"] == "defaulted").mean()

    # 2. Income Verification & Discrepancy Audit
    missing_docs_count = df["documented_monthly_income"].isna().sum()
    missing_docs_pct = (missing_docs_count / total_n) * 100

    # For applicants with docs: calculate stated / documented ratio
    has_docs_df = df[df["documented_monthly_income"].notna() & (df["documented_monthly_income"] > 0)].copy()
    has_docs_df["income_ratio"] = has_docs_df["stated_monthly_income"] / has_docs_df["documented_monthly_income"]

    # Observed 3x+ discrepancy (as described in prose) vs 2.5x-5x (as coded in generator)
    ratio_gte_3x_count = (has_docs_df["income_ratio"] >= 3.0).sum()
    ratio_gte_2_5x_count = (has_docs_df["income_ratio"] >= 2.5).sum()

    # 3. Employment Status Breakdown
    emp_summary = []
    for status, group in df.groupby("employment_status"):
        res_group = group[group["actual_outcome"] != "ongoing"]
        n_total = len(group)
        n_resolved = len(res_group)
        defaults = (res_group["actual_outcome"] == "defaulted").sum()
        default_rate = defaults / n_resolved if n_resolved > 0 else 0
        rule_approved = (group["rule_based_decision"] == "approved").mean()
        rule_review = (group["rule_based_decision"] == "flagged_for_review").mean()
        rule_denied = (group["rule_based_decision"] == "denied").mean()
        emp_summary.append({
            "employment_status": status,
            "total_applicants": n_total,
            "resolved_loans": n_resolved,
            "defaults": defaults,
            "default_rate": default_rate,
            "rule_approval_rate": rule_approved,
            "rule_review_rate": rule_review,
            "rule_denial_rate": rule_denied,
        })
    emp_df = pd.DataFrame(emp_summary)

    # 4. Generate Diagnostic Plot 1: Outcomes & Decisions by Employment
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette = {"employed": "#2b5c8f", "self_employed": "#d95f02", "unemployed": "#7570b3"}

    # Left: Empirical Default Rate among resolved loans
    sns.barplot(
        data=emp_df,
        x="employment_status",
        y="default_rate",
        hue="employment_status",
        palette=palette,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Empirical Default Rate by Employment (Resolved Loans)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Default Rate", fontsize=11)
    axes[0].set_xlabel("Employment Status", fontsize=11)
    axes[0].set_ylim(0, 0.6)
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(
            f"{height * 100:.1f}%",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            xytext=(0, 4),
            textcoords="offset points",
        )

    # Right: Rule-Based Decision Distribution by Employment
    decision_df = (
        df.groupby(["employment_status", "rule_based_decision"])
        .size()
        .unstack(fill_value=0)
        .apply(lambda r: r / r.sum(), axis=1)[["approved", "flagged_for_review", "denied"]]
    )
    decision_df.plot(
        kind="bar",
        stacked=True,
        color=["#2ca02c", "#ff7f0e", "#d62728"],
        ax=axes[1],
        rot=0,
    )
    axes[1].set_title("Rule-Based Decision Breakdown by Employment", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Proportion of Group", fontsize=11)
    axes[1].set_xlabel("Employment Status", fontsize=11)
    axes[1].legend(title="Rule Decision", loc="upper right")
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plot1_path = os.path.join(plot_dir, "eda_outcomes_and_employment.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()

    # 5. Generate Diagnostic Plot 2: Missing Documentation & Income Discrepancies
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Num Docs vs Missing Documented Income
    doc_cross = pd.crosstab(df["num_documents_submitted"], df["documented_monthly_income"].isna())
    doc_cross.columns = ["Docs Documented", "Docs Missing"]
    doc_cross.plot(kind="bar", stacked=True, color=["#1f77b4", "#e377c2"], ax=axes[0], rot=0)
    axes[0].set_title("Document Submission vs. Missing Income Record", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Documents Submitted", fontsize=11)
    axes[0].set_ylabel("Number of Applicants", fontsize=11)
    axes[0].legend(title="Income Field Status")

    # Right: Stated to Documented Income Ratio
    sns.histplot(
        has_docs_df["income_ratio"],
        bins=40,
        kde=True,
        color="#17becf",
        ax=axes[1],
    )
    axes[1].axvline(1.1, color="green", linestyle="--", label="10% Tolerance Boundary (1.1x)")
    axes[1].axvline(2.5, color="red", linestyle="--", label="Misrepresentation Cluster (2.5x - 5.0x)")
    axes[1].set_title("Stated / Documented Income Ratio (When Provided)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Ratio (Stated Monthly Income / Documented Monthly Income)", fontsize=11)
    axes[1].set_ylabel("Applicant Count", fontsize=11)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plot2_path = os.path.join(plot_dir, "eda_missing_and_discrepancy.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()

    eda_results = {
        "total_applicants": total_n,
        "resolved_loans": resolved_n,
        "outcome_counts": outcome_counts,
        "outcome_props": outcome_props,
        "resolved_default_rate": float(resolved_default_rate),
        "missing_docs_count": int(missing_docs_count),
        "missing_docs_pct": float(missing_docs_pct),
        "ratio_gte_3x_count": int(ratio_gte_3x_count),
        "ratio_gte_2_5x_count": int(ratio_gte_2_5x_count),
        "employment_summary": emp_df.to_dict(orient="records"),
        "plots": [plot1_path, plot2_path],
    }

    print("--- EDA Summary ---")
    print(f"Total Applicants: {total_n} | Resolved Loans: {resolved_n} ({resolved_default_rate*100:.1f}% default rate)")
    print(f"Outcome Distribution: {outcome_counts}")
    print(f"Missing Documented Income: {missing_docs_count} ({missing_docs_pct:.1f}%)")
    print(f"Income Discrepancy >= 2.5x: {ratio_gte_2_5x_count} | >= 3.0x: {ratio_gte_3x_count}")
    print("\nEmployment Group Audit (Resolved):")
    print(emp_df[["employment_status", "total_applicants", "default_rate", "rule_approval_rate"]].to_string(index=False))
    print(f"\nPlots saved to: {plot1_path}, {plot2_path}")

    return eda_results


if __name__ == "__main__":
    run_eda()
