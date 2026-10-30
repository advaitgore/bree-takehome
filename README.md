# Bree ML Take-Home: Loan Underwriting Prototype

A reproducible, modular Python pipeline that trains, evaluates, and audits an ML underwriting model against Bree's legacy rule-based scoring system. Emphasizes **methodological consistency, honest economic modeling, and grounded reasoning**.

---

## Quickstart & Reproduction

**Environment Note**: Tested with Python 3.12+ (Miniconda). Pinned dependencies (including `shap==0.54.1`) require Python 3.12 or newer.

```bash
# 1. Install locked dependencies
pip install -r requirements.txt

# 2. Run the end-to-end pipeline (data generation, EDA, training, evaluation, fairness, SHAP)
python run_pipeline.py

# 3. Run consistency test suite
pytest tests/
```

- **Plots**: `plots/`
- **Machine-readable metrics**: `output/metrics.json` and `output/fairness.json`
- **Data**: `data/loan_applications.csv` (synthetic dataset, seed 42, $N=2{,}000$)

---

## Executive Summary

### 1. Data Handling & Generator Audit

* **Default Prevalence**: The prompt mentions "~14% default", but the actual generator yields **27.25% overall** (545 of 2,000) and **29.7% on resolved loans** (545 of 1,836).
* **Missing Documentation & Discrepancy**: 283 applicants (14.1%) submit zero documents, causing `documented_monthly_income` to be null. We impute this with 0.0 and flag it via `is_doc_missing = 1`. In the full dataset, among the 1,717 applications with documentation, 85 show an income discrepancy ratio $\ge 2.5\times$ and 56 show $\ge 3.0\times$. The rule's 10% tolerance boundary on stated income corresponds to an upper ratio of $1 / 0.9 \approx 1.111$.
* **Ongoing Applications**: In the generator script, `ongoing` status is assigned independently after latent outcomes are generated, making label availability **Missing Completely At Random (MCAR) by construction**. Excluding ongoing loans ($N=164$) from supervised training causes zero selection bias in expectation. In production, excluding unresolved loans can introduce seasoning or selection bias, depending on the outcome horizon and cohort construction.
* **Why No Survival Analysis**: `days_to_default` was generated as uniform random integers (`randint(15, 180)`) independent of risk, with no observation windows or censoring dates for non-defaulters. Fitting survival models on this variable is invalid.
* **Leakage Controls & Single Frozen Split**: Resolved applications ($N=1{,}836$) are split into a stratified 75/25 holdout (1,377 train / 459 test, seed 42). Target (`actual_outcome`), proxy duration (`days_to_default`), identifiers, and rule decisions are strictly excluded from features. For logistic regression, preprocessing is cross-validated inside the hyperparameter grid search (`GridSearchCV`) to avoid internal scaling leakage across CV folds.

---

### 2. Candidate Models, Feature Importance & Operating Threshold

* **Architectures Tested**:
  1. *Regularized Logistic Regression*: L2 penalty tuned via 5-fold CV ($C = 0.1668$).
  2. *Shallow XGBoost*: Constrained to `max_depth=3`, `subsample=0.85`, `colsample_bytree=0.85`, `min_child_weight=3`, `learning_rate=0.05`, 60 estimators.
* **Feature Importance (TreeSHAP)**:
  Examining global mean absolute SHAP values on the holdout (`plots/shap_feature_importance.png`) reveals that default risk is driven primarily by:
  1. `loan_to_income_ratio` (0.508 mean |SHAP| in log-odds)
  2. `doc_income_imputed` (0.283 log-odds)
  3. `bank_has_overdrafts` (0.201 log-odds)
  4. `income_discrepancy_ratio` (0.099 log-odds)
  5. `bank_ending_balance` (0.097 log-odds)
  These attributions verify that the tree model learned risk directly from cash-flow capacity and liquidity distress rather than spurious noise.
* **Single Shared Operating Threshold (0.35)**:
  Both ML models are evaluated at an illustrative, non-optimized decision threshold of **0.35**, approximating the training default prevalence (~30%). This threshold is defined once in `src/models.py` and passed to all downstream evaluation, fairness, and explainability modules. It is **not an EV-optimized threshold** (under $M=\$100, L=\$1{,}000$, a pure risk-neutral expected value policy would approve only when $p < \frac{100}{100+1000} \approx 9.1\%$).

---

### 3. Model Performance & Economic Comparison

#### Policy Mappings
For binary evaluation, **Deny-Only** treats legacy denials as positive predictions (flagged for default) and groups reviews with approvals; **Review-or-Deny** treats both reviews and denials as positive predictions. These are analytical mappings, not observed final outcomes of manual review.

#### Holdout Test Performance ($N=459$: 136 defaults, 323 repaid)

| Policy / Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | FNR | Defaults Caught (TP) | Good Denied (FP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deny-Only (Rule)** | 0.725 | 0.549 | **0.627** | 0.309 | 0.414 | **0.077** | 0.691 | 42 | **25** |
| **Review-or-Deny (Rule)** | 0.725 | 0.549 | 0.409 | **0.765** | 0.533 | 0.464 | **0.235** | **104** | 150 |
| **Logistic Regression (0.35)** | 0.727 | 0.526 | 0.475 | 0.566 | 0.517 | 0.263 | 0.434 | 77 | 85 |
| **XGBoost (0.35)** | **0.761** | **0.575** | 0.553 | 0.618 | **0.583** | 0.211 | 0.382 | 84 | 68 |

*XGBoost achieved the highest ROC-AUC (0.761 vs. 0.725) and PR-AUC (0.575 vs. 0.549). Deny-Only achieved highest precision (0.627) and lowest good-borrower rejection.*

#### Illustrative Contribution Comparison

Scenario calculation assuming contribution margin $M=\$100$ per approved good loan, principal loss $L=\$1{,}000$ per approved defaulted loan, zero recovery, and no volume utility. Rejection yields \$0.

| Policy | Good Approved (TN) | Defaults Approved (FN) | Illustrative Contribution |
| :--- | :---: | :---: | :---: |
| **Approve Everyone** (reference) | 323 | 136 | −$103,700 |
| **Deny-Only (Rule)** | 298 | 94 | −$64,200 |
| **Logistic Regression (0.35)** | 238 | 59 | −$35,200 |
| **XGBoost (0.35)** | 255 | 52 | −$26,500 |
| **Review-or-Deny (Rule)** | 173 | 32 | **−$14,700** |
| **Reject Everyone** (reference) | 0 | 0 | **$0** |

#### Economic Analysis & Tradeoffs
1. **XGBoost vs. Deny-Only Rule**: XGBoost catches $+42$ additional defaults while denying $+43$ additional good applicants. Under the illustrative $L/M=10$ scenario, XGBoost improves contribution relative to Deny-Only. The improvement remains positive whenever $L/M > 43/42 \approx 1.02$.
2. **Review-or-Deny vs. XGBoost**: Review-or-Deny catches $+20$ additional defaults but denies $+82$ good applicants ($\frac{L}{M} > \frac{82}{20} = 4.10$). Because our assumed $\frac{L}{M} = 10.0 > 4.10$, **Review-or-Deny exhibits the highest contribution of any active policy in this scenario**. XGBoost at 0.35 was chosen to balance precision and recall, not to maximize expected value under an extreme loss-ratio assumption.
3. **Reject-Everyone Reference**: Under this simplified P&L, all active policies produce negative net contribution due to the high test default prevalence (29.7%) and severe loss assumption. Rejecting everyone generates \$0, but ceases all lending operations.
4. **Governance**: Credit Risk leadership and the CRO decide the threshold based on risk appetite, target loss rates, and capital constraints.

---

### 4. Fairness & Employment Group Audit

#### Root Causes in the Rule Baseline
- **Direct Penalty**: Self-employed applicants receive 60/100 on employment score vs. 100/100 for employed. Weighted at 15%, this is an automatic **6.0-point total score penalty**.
- **Indirect Deposit Penalty**: Consistent deposits contributes 30 component points $\times$ 20% weight = **6.0 total points**. While employed applicants have an 80% consistency rate vs. 40% for self-employed, **deposit consistency does not appear in the true default generation equation**.
- *(Note: Account balances are generated independently of employment; observed group balance differences reflect sample noise, not designed mechanisms).*

#### Audit Across Groups (XGBoost at 0.35 on Test Set $N=459$)

Per-group confusion counts sum exactly to the overall model performance counts: $\sum \text{TP} = 37 + 25 + 22 = 84$; $\sum \text{FP} = 40 + 22 + 6 = 68$.

| Group | Count | Default Rate | Approval Rate | Good Denied Rate (FPR) | Default Catch Rate (Recall) | TP | FP | TN | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Employed** | 271 | 25.8% | 71.6% | 19.9% | 52.9% | 37 | 40 | 161 | 33 |
| **Self-Employed** | 140 | 27.1% | 66.4% | 21.6% | 65.8% | 25 | 22 | 80 | 13 |
| **Unemployed** | 48 | 58.3% | 41.7% | 30.0% | 78.6% | 22 | 6 | 14 | 6 |

#### Self-Employed to Employed Approval Ratios (4/5ths reference = 0.80)
- **Deny-Only (Rule)**: $80.0\% / 93.4\% = \mathbf{0.857}$
- **Review-or-Deny (Rule)**: $37.1\% / 56.5\% = \mathbf{0.658}$ (disparity driven by conservative review gating)
- **XGBoost (Full Model)**: $66.4\% / 71.6\% = \mathbf{0.928}$
- **XGBoost (Blinded: No Employment Feature)**: $65.7\% / 71.6\% = \mathbf{0.918}$

#### Ablation: Removing Employment Status & Recommendation
Training a blinded model without `employment_status` on the identical frozen split yields:
- 21 prediction disagreements out of 459 test instances (4.6%).
- Maximum absolute probability difference: $0.0940$.
- ROC-AUC: $0.7606$ (full) vs. $0.7557$ (blinded).
- **Recommendation**: Removing employment status did not improve the measured approval ratio in this experiment (0.918 vs. 0.928) and slightly reduced observed ROC-AUC (0.7557 vs. 0.7606). I would not claim either variant is demonstrably fairer from this small holdout. Employment's explicit use should remain a policy-review question, with group outcomes monitored under either choice.

---

### 5. Explainability & Production Readiness

#### Adverse Action Reason Prototype
Using TreeSHAP in log-odds space (base value: $-0.8706$, additivity MAE: $0.000000$), positive values reflect factors increasing predicted default log-odds. Reasons are phrased neutrally without asserting unverified rule violations:
- *Case 1 (True Positive illustration)*: `APP-0311` (Employed, Loan \$1,500, Balance \$0, Defaulted). Model prob: 0.56 (Denied). Top risk factors: Documented income level (+0.63 log-odds), Loan-to-income ratio (+0.37 log-odds), Bank ending balance (+0.20 log-odds).
- *Case 2 (False Positive illustration)*: `APP-1344` (Self-employed, Loan \$1,500, Balance \$1,880, Repaid). Model prob: 0.38 (Denied). Top risk factors: Loan-to-income ratio (+0.35 log-odds), Documented income level (+0.28 log-odds). Illustrates the cost of threshold selection on borderline applicants.
- *Case 3 (Fairness recovery illustration)*: `APP-1849` (Self-employed, Loan \$1,000, Balance \$2,320, Repaid). Rule: Flagged for review (score 53.4); Model: Approved (prob 0.26).
- *Regulatory Note*: These represent prototype internal reviewer codes, not legally compliant FCRA/ECOA adverse-action notices.

#### Macroeconomic Shift / Recession
- **Label Shift**: Assumes class-conditional feature distributions remain invariant ($P_{\text{new}}(X \mid Y) = P_{\text{base}}(X \mid Y)$) while default prevalence $P(Y)$ increases. Under this assumption, predictions can be recalibrated via log-odds prior shift correction:
  $$\text{logit}(P_{\text{new}}) = \text{logit}(P_{\text{base}}) + \ln\left(\frac{\pi_{\text{new}}}{1 - \pi_{\text{new}}}\right) - \ln\left(\frac{\pi_{\text{base}}}{1 - \pi_{\text{base}}}\right)$$
- If the recession induces covariate shift or concept drift, prior correction alone is insufficient. Unconstrained decision trees do not guarantee monotonic risk responses to balance declines without explicit monotonicity constraints.

#### Deployment Checklist (Day-1 Failure Modes) & Limitations
1. **Reject Inference**: Production historical data only has repayment labels for *approved* applicants. Retraining solely on approved loans introduces severe truncation bias.
2. **Feature Parity**: Verify bank transaction aggregations (e.g. deposit consistency, overdraft definitions) match production core banking APIs.
3. **Shadow Mode**: Deploy the ML pipeline in silent shadow execution for 60–90 days to evaluate calibration and drift (monitoring Population Stability Index on key cash-flow features, flagging $PSI > 0.25$) before cutover.
4. **Scope & Limitations**: Results are point estimates from one synthetic holdout; probability calibration and uncertainty in model differences have not been fully assessed.
