# Bree ML Take-Home: Loan Underwriting Prototype

A reproducible, modular Python pipeline that trains, evaluates, and audits an ML underwriting model against Bree's legacy rule-based system. The emphasis is on **honest reasoning, leakage prevention, and clearly qualified conclusions**—not superficial sophistication.

---

## Quickstart

```bash
pip install -r requirements.txt
python run_pipeline.py        # generates data, plots, and prints all results
```

All plots are saved to `plots/`. The synthetic dataset is committed at `data/loan_applications.csv`.

**Environment note**: Tested with Python 3.12.9 (Miniconda). The `run_pipeline.py` script is the single reproducible entry point.

---

## Written Summary

### 1. Data Handling and Generator Audit

#### Generator Audit vs. Prompt Discrepancies

Two notable differences exist between the prompt's prose description and the actual generator:

1. **Default prevalence**: The prompt states "~14% default." The seed-42 dataset yields **27.25% overall** (545/2,000) and **29.7% of resolved loans** (545/1,836). The prose likely describes an overall rate including ongoing loans under different assumptions; the generator produces a materially higher resolved-loan rate.

2. **Income discrepancy magnitude**: The generator assigns documented income as 20%–40% of stated income for the 5% latent misrepresentation group, producing discrepancy ratios of **2.5×–5.0×**, not exactly 3× as the prose implies. Additionally, 14.1% of applicants (283 of 2,000) submit zero documents; their membership in the misrepresentation group cannot be observed. On the resolved test set: **85 applications show discrepancy ≥ 2.5×** and **56 show ≥ 3.0×**.

#### Missing Documentation

`documented_monthly_income` is null for all 283 applicants with `num_documents_submitted == 0`. Rather than dropping rows or imputing with the training mean, we create a binary `is_doc_missing` indicator and set `doc_income_imputed = 0.0`. When documents are missing, `income_discrepancy_ratio` is set to a neutral 1.0. See `plots/eda_missing_and_discrepancy.png`.

#### Ongoing Applications

**In this synthetic dataset**: `ongoing = random() < 0.08` is drawn *after* latent default status is determined and independently of all applicant characteristics. Label availability is **Missing Completely At Random (MCAR) by construction**. Excluding ongoing applications from supervised training and evaluation introduces zero selection bias in expectation.

**In real-world underwriting**: Excluding unresolved loans can introduce seasoning or selection bias when label availability depends on cohort age or risk profile. Appropriate treatment requires a defined outcome horizon and valid follow-up information. Newer cohorts have not had sufficient exposure time to mature, so excluding them selectively over-indexes on older, seasoned loans or early defaults. We document this caveat but do not attempt to model it here because the synthetic data does not exhibit it.

**Why no survival analysis**: `days_to_default` is assigned via `randint(15, 180)` independently of borrower risk. There are no censoring times for non-defaulted loans. Applying survival modeling to this variable would be scientifically invalid.

#### Evaluation Protocol and Leakage Controls

Resolved applications (N=1,836) are filtered and frozen into a stratified 75/25 train/test split (1,377 train / 459 test, `random_state=42`, stratified on outcome). The split is created before any model is fit and held constant throughout. The target (`actual_outcome`), leakage columns (`days_to_default`, `applicant_id`), and rule-system outputs (`rule_based_score`, `rule_based_decision`) are strictly excluded from the predictor matrix via an explicit allowlist. All preprocessing (scaling, encoding) is fitted exclusively on training indices and then applied to test indices.

Smoke tests performed: pipeline executes without error; dataset row count = 2,000 (verified); test set size = 459 (verified); no target column present in feature matrix (verified by allowlist); all plots written to disk.

---

### 2. Model Selection and Feature Attributions

#### Architecture Choices

The generator computes default probability as a logistic function of a linear risk composite:

$$\text{default\_prob} = \frac{1}{1 + e^{-5 \times (\text{real\_risk} - 0.45)}}$$

Given this functional form, two architectures were tested:

1. **Regularized Logistic Regression** — a transparent linear model whose decision surface closely matches the generator's structure. L2 penalty selected via 5-fold cross-validation (`LogisticRegressionCV`). Useful as a linear baseline and for interpretability comparison.

2. **Shallow XGBoost** — constrained to `max_depth=3`, `subsample=0.85`, `learning_rate=0.05`, 60 estimators. Shallow depth limits overfitting on 1,377 training rows while allowing the model to capture threshold-like non-linearities (e.g., `loan_amount ≤ 0.3 × stated_income`).

Both models use identical features and preprocessing, evaluated on identical test indices.

#### Feature Importance (TreeSHAP)

SHAP values are in **log-odds units** relative to the TreeExplainer base value. Positive attribution increases predicted default probability. The top drivers on the holdout set (`plots/shap_feature_importance.png`):

| Rank | Feature | Mean \|SHAP\| |
|------|---------|---------------|
| 1 | `loan_to_income_ratio` | 0.508 |
| 2 | `doc_income_imputed` | 0.283 |
| 3 | `bank_has_overdrafts` | 0.201 |
| 4 | `income_discrepancy_ratio` | 0.099 |
| 5 | `bank_ending_balance` | 0.097 |

These align closely with the generator's true risk factors, providing a consistency check that the model is learning genuine signal rather than proxies or artifacts.

---

### 3. Results, Baseline Comparison, and Business Tradeoffs

#### Holdout Performance (N = 459 resolved applications; 136 defaults, 323 repaid)

| Policy / Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | FNR | Defaults Caught | Good Denied |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Deny-Only (Rule)** | 0.725 | 0.549 | **0.627** | 0.309 | 0.414 | **0.077** | 0.691 | 42 | 25 |
| **Review-or-Deny (Rule)** | 0.725 | 0.549 | 0.409 | **0.765** | 0.533 | 0.464 | 0.235 | **104** | 150 |
| **Logistic Regression (0.35)** | 0.727 | 0.526 | 0.481 | 0.551 | 0.514 | 0.251 | 0.449 | 75 | 81 |
| **XGBoost (0.35)** | **0.761** | **0.575** | 0.568 | 0.581 | **0.575** | 0.186 | 0.419 | 79 | 60 |

*(See `plots/roc_pr_curves.png` and `plots/confusion_matrices.png`.)*

**Honest interpretation**: XGBoost achieved the strongest observed ranking performance on this holdout (ROC-AUC 0.761 vs. 0.725). The Deny-Only rule achieves the highest precision (0.627) and lowest good-applicant rejection rate. Policy preference depends on the assumed economics, how "flagged for review" loans are handled operationally, and volume constraints — not AUC alone.

#### Threshold Selection

The 0.35 threshold was selected as a **balanced operating point approximating the training default prevalence (~30%)**, not via cost-sensitive optimization. It was chosen before inspecting test-set results. Both ML models use the same threshold for comparability.

Under a pure expected-value criterion, approval is favorable when the predicted probability $p < M / (M + L)$. With $M = \$100$ and $L = \$1{,}000$, that implies a threshold near **9%–13%** — well below 0.35. The 0.35 threshold therefore prioritizes recall/precision balance over expected-value maximization. Proper cost-sensitive threshold selection belongs within the development procedure on a validation set, not after viewing test results.

#### Illustrative Policy Contribution (Scenario Calculation)

**Assumptions (explicitly stated; not validated against real lender economics):**
- $M = \$100$ per approved good loan (origination fees and net interest, illustrative)
- $L = \$1{,}000$ per approved defaulted loan (principal charge-off, illustrative)
- Constant per loan; recoveries, review costs, and funding costs omitted
- All rejected good applicants treated as permanently lost (no off-take modeled)

| Policy | Good Approved | Defaults Approved | Illustrative Contribution |
|:---|:---:|:---:|:---:|
| Approve Everyone (reference) | 323 | 136 | **−\$103,700** |
| Deny-Only (Rule) | 298 | 94 | −\$64,200 |
| Logistic Regression (0.35) | 242 | 61 | −\$36,800 |
| XGBoost (0.35) | 263 | 57 | −\$30,700 |
| Review-or-Deny (Rule) | 173 | 32 | −\$14,700 |
| Reject Everyone (reference) | 0 | 0 | **\$0** |

These are retrospective scenario calculations using holdout counts — not actual lender profit.

**Key qualifications**:

- **XGBoost (0.35) is not the best displayed policy under these illustrative assumptions.** Review-or-Deny catches +25 more defaults than XGBoost while denying +90 more good applicants. This yields a break-even L/M of **3.6×** — below the assumed 10×. Under our illustrative economics, Review-or-Deny has the highest contribution of any active policy.
- **XGBoost improves contribution relative to the Deny-Only baseline** (from −\$64,200 to −\$30,700) — a meaningful improvement that does not require a specific L/M assumption to validate.
- **No policy produces positive net contribution** under these assumptions. The reference "reject everyone" policy at \$0 dominates all active policies, which is a well-known artifact of simplified credit P&L models that ignore loan volume, credit access obligations, and customer lifetime value. Use the reject-all reference to expose this mismatch rather than as a policy recommendation.
- Absolute profitability is not established by this analysis. Real lender economics require validated margins, recoveries, cost of funds, and operational costs.

#### Business Tradeoff: Who Decides?

**Risk leadership** (Credit Risk Committee, CRO) defines the loss tolerance and target approval volume. The ML engineer's responsibility is to surface the Pareto operating curve across all thresholds — not to pick the operating point. The engineer should present the full contribution table above and let stakeholders choose based on their risk appetite, capital constraints, and regulatory environment.

---

### 4. Fairness and Employment Group Disparities

#### What the Generator Reveals

The rule system imposes two compounding penalties on self-employed applicants:

1. **Direct penalty**: `emp_score` assigns self-employed applicants 60/100 points vs. 100/100 for employed. With a 15% weight, this is a **−6.0 point penalty on the total score** out of 100.

2. **Indirect proxy**: `bank_has_consistent_deposits` is generated at 80% frequency for employed vs. 40% for self-employed. The rule assigns 30 component points (×15% = 4.5 total points) to deposit consistency — **even though deposit consistency does not appear in the generator's true default probability equation.** This creates an additional, uncompensated penalty.

Other generated differences (income distributions, balance levels) may also contribute to the observed approval gap. A formal decomposition would require controlling for those factors individually.

#### Full Employment Group Audit (XGBoost at 0.35)

| Group | Count | Default Rate | Approval Rate | Good Denied Rate (FPR) | Default Catch Rate |
|:---|:---:|:---:|:---:|:---:|:---:|
| Employed | 271 | 25.8% | 70.5% | 21.9% | 51.4% |
| Self-employed | 140 | 27.1% | 67.1% | 21.6% | 63.2% |
| Unemployed | 48 | 58.3% | 45.8% | 30.0% | 71.4% |

| Group | Deny-Only Approval Rate | Review-or-Deny Approval Rate |
|:---|:---:|:---:|
| Employed | 93.4% | 50.6% |
| Self-employed | 80.0% | 40.0% |
| Unemployed | 56.3% | 18.8% |

**Self-employed/employed approval ratios** (4/5ths rule benchmark: 0.80):
- Deny-Only (Rule): **0.857** ✓
- Review-or-Deny (Rule): **0.658** ✗ (disparate impact)
- XGBoost Full Model: **0.953** ✓
- XGBoost Blinded (no employment feature): **0.953** ✓

**Interpretation**: On this evaluated sample, the selected ML model reduced the observed employed/self-employed approval-rate gap, and good-applicant rejection rates are similar across groups (21.9% vs. 21.6%). This does not establish fairness across all groups, metrics, or future populations.

The unemployed group (N=48 in test) has a substantially higher default rate (58.3% vs. ~26% for other groups) and correspondingly higher denial rates. The small sample size limits conclusions for this group.

#### Ablation: Employment Feature Removal

Training XGBoost without `employment_status` as an input feature produces identical approval rates and predictions on this test set. The model appears to price risk through cash-flow indicators (overdrafts, balances, income ratios) without needing the employment label.

**Qualified claim**: Removing employment status eliminates its explicit use as a model input. The employment-excluded model retained similar observed approval parity in this experiment. However, proxy effects (deposit consistency, balance levels) are correlated with employment status and may carry residual disparate impact. Full proxy analysis would require additional tools.

**Recommendation**: Remove `employment_status` from the model. It contributes negligible predictive lift (identical ROC-AUC), and removing it eliminates potential disparate-treatment claims while making the self-employed approval gap harder to attribute directly to the model.

---

### 5. Explainability, Regulatory Audit, and Production Risk

#### Regulatory Audit: "Why was this applicant denied?"

The rule-based system is fully inspectable: score components and thresholds are deterministic and documented. ML predictions require post-hoc interpretation.

We implemented an **illustrative adverse-action reason prototype** (`src/explainability.py`) using TreeSHAP waterfall values. For each denied applicant, we extract the top positive SHAP contributors (positive = increases predicted default log-odds relative to the base value) and map them to human-readable phrases:

- *APP-0311* (Denied, predicted prob 0.56): "Low or unverified documented income (+0.63 log-odds)", "High loan-to-income ratio (+0.37 log-odds)", "Low ending bank balance (+0.20 log-odds)".
- *APP-1849* (Self-employed): Flagged for review by rule system (score 53.4) but approved by XGBoost (prob 0.26) due to healthy bank balance and no overdrafts; subsequently repaid.

**Important qualification**: These are illustrative reviewer reason codes, not legally compliant adverse-action notices. FCRA and ECOA adverse-action requirements mandate specific language, a defined process, and legal/compliance review. This prototype demonstrates the technical capability; legal review would be required before any production use.

#### Macroeconomic Shift / Recession

If default prevalence increases substantially (e.g., due to recession) and applicant financial characteristics deteriorate:

- **Rule system**: Deteriorating bank balances reduce the account-stability component score; applicants with `balance < threshold` or overdraft flags receive lower scores and may cross decision thresholds. The rule system does not automatically recalibrate — it applies the same weights under different distributional conditions.
- **ML model**: Predicted probabilities increase as features deteriorate, which is the desired direction. However, the model is not calibrated for the new distribution. The prior-odds log-odds correction formula provides a proposed recalibration approach:

$$\text{logit}(P_{\text{new}}) = \text{logit}(P_{\text{base}}) + \ln\!\left(\frac{\pi_{\text{new}}}{1 - \pi_{\text{new}}}\right) - \ln\!\left(\frac{\pi_{\text{base}}}{1 - \pi_{\text{base}}}\right)$$

This correction is **contingent on the label-shift assumption** — that the feature-conditional default probability structure is unchanged, and only the marginal default rate has shifted. If the recession also changes which features predict default (a covariate shift or concept drift), this correction is insufficient. Neither policy has demonstrated recession robustness on real data.

#### Day-1 Failure Modes if Deployed Tomorrow

The first and most serious failure mode is **synthetic-to-real mismatch**: the model learned from a generator with known, simple structure. Real applicant data involves messier distributions, additional features, data pipeline errors, and more complex relationships. Before changing any decisions, I would:

1. Validate feature definitions against real data pipelines (are `bank_ending_balance`, `income_discrepancy_ratio`, and `bank_has_overdrafts` computed the same way in production?).
2. Evaluate calibration on real, appropriately matured cohorts with confirmed outcomes.
3. Run shadow scoring — log model outputs for 60–90 days alongside the existing policy without changing decisions.
4. Assess the **selective-label / reject-inference problem**: unlike this synthetic dataset where we observe outcomes for all 2,000 applicants (including those the rule model denied), real historical data only provides repayment outcomes for previously approved applicants. Denied applicants never generate default data, which biases training if the model is retrained on real historical data.
5. Monitor input feature distributions weekly (e.g., overdraft frequency, balance levels) relative to training baseline — PSI > 0.25 on key features triggers a formal review.

---

## Repository Structure

```
├── data/
│   └── loan_applications.csv          # Synthetic dataset (seed 42, N=2,000)
├── plots/
│   ├── eda_missing_and_discrepancy.png
│   ├── eda_outcomes_and_employment.png
│   ├── roc_pr_curves.png
│   ├── confusion_matrices.png
│   ├── fairness_disparate_impact.png
│   └── shap_feature_importance.png
├── src/
│   ├── __init__.py
│   ├── generate_data.py               # Exact generator logic preserved
│   ├── eda.py                         # Diagnostic EDA and generator audit
│   ├── features.py                    # Predictor allowlist and leakage guards
│   ├── models.py                      # Rule baseline, Logistic Regression, XGBoost
│   ├── evaluate.py                    # Metrics, confusion matrices, contribution table
│   ├── fairness.py                    # Employment group audit and ablation
│   └── explainability.py              # TreeSHAP and illustrative reason codes
├── run_pipeline.py                    # Single reproducible entry point
├── requirements.txt
└── README.md
```

---

## Final Assessment

XGBoost achieved the strongest observed ranking performance on the reported holdout (ROC-AUC 0.761 vs. 0.725 for the rule baseline). That does not by itself establish the best lending policy.

Under the illustrative economic assumptions used here, the Review-or-Deny rule policy produces the highest scenario contribution among active policies — XGBoost at threshold 0.35 was selected for F1-balanced performance, not economic dominance. No active policy produces a positive net contribution under these simplified assumptions, which reflects the structural tension between loss avoidance and loan volume.

Deployment decisions require validated probabilities on real data, explicit and defensible economic assumptions, consistent handling of flagged-for-review loans, a solution to the reject-inference problem, operational monitoring infrastructure, and legal/compliance review of the reason-code system. The findings here are a starting point for that conversation, not a production recommendation.
