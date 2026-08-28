# Sprint 5 — Data Cleaning & Data Preprocessing for AI/ML Engineers

A complete, hands-on preprocessing series converting a raw, real-world dataset into a
clean, validated, transformed, and ML-ready dataset — continuing directly from the
Sprint 4 EDA findings on the same Telco Customer Churn dataset.

## Objective

A Machine Learning model cannot compensate for poor-quality data. This sprint follows
one guiding principle across all 17 notebooks:

**Identify → Analyze → Clean → Transform → Validate → Prepare for Machine Learning**

Every preprocessing decision is documented using a consistent seven-part framework:
**Problem → Analysis → Technique Selected → Reason → Implementation → Result → Impact.**

## Dataset

Continuing directly from Sprint 4: **Telco Customer Churn** (7,043 customers × 21
columns, IBM sample data, telecom/customer-churn domain, target `Churn`). Every
data-quality issue this sprint addresses was first identified and documented during
Sprint 4's EDA — this sprint's job is to actually fix them, with the reasoning behind
each fix made explicit.

**Real issues identified in Sprint 4 and resolved in this sprint:**
- `TotalCharges` stored as text with 11 disguised missing values (Notebook 2–3)
- 112 genuine multivariate billing-history outliers (Notebook 6 — retained, not removed)
- 16 categorical columns requiring encoding (Notebook 7)
- Moderately imbalanced target, 73.46%/26.54% (Notebook 11)
- Multicollinearity between `TotalCharges` and `tenure`×`MonthlyCharges` (VIF ≈ 8.1,
  addressed via feature selection guidance, Notebook 10)

## Learning Methodology

Every concept follows: **Understand** (own words) → **Demonstrate** (real-world/business/
AI-ML example, explaining what problem the technique solves) → **Implement** (Python,
explaining what the code does, why the technique was selected, before/after state, why
it's appropriate, and the ML impact).

## Notebooks

| # | Notebook | Covers |
|---|---|---|
| 01 | `01_Data_Preprocessing_Basics.ipynb` | What/why preprocessing, raw vs. clean data, the 5 data-quality dimensions, data leakage (demonstrated numerically), train/val/test roles, the full sprint roadmap |
| 02 | `02_Data_Type_Handling.ipynb` | Identifying/converting types, `astype()` vs `to_numeric()`, the `TotalCharges` fix, automated type-issue detection |
| 03 | `03_Missing_Value_Handling.ipynb` | MCAR/MAR/MNAR classification, 10+ imputation techniques compared and mostly *rejected* in favor of a justified constant fill |
| 04 | `04_Duplicate_Data.ipynb` | Exact vs. partial duplicates; 42 near-duplicate rows investigated and correctly **not** removed |
| 05 | `05_Data_Validation.ipynb` | 10 validation categories, each implemented as a real Python check; a consolidated `validate_dataset()` function |
| 06 | `06_Outlier_Treatment.ipynb` | IQR/Z-score/percentile methods, winsorization/capping mechanics demonstrated, and the reasoned decision to retain all 112 multivariate outliers |
| 07 | `07_Categorical_Encoding.ipynb` | Label/ordinal/one-hot/frequency/target encoding compared; the false-order risk of label-encoding a nominal variable shown directly |
| 08 | `08_Feature_Scaling.ipynb` | StandardScaler/MinMaxScaler/RobustScaler/MaxAbsScaler compared; outlier sensitivity demonstrated with an injected synthetic outlier |
| 09 | `09_Data_Transformation.ipynb` | Log/sqrt/Box-Cox/Yeo-Johnson compared head-to-head on `TotalCharges`'s skew; Box-Cox's failure on zero values shown directly |
| 10 | `10_Feature_Selection.ipynb` | Correlation, variance threshold, univariate F-test, mutual information, RFE, and Random Forest importance — cross-method agreement analysis |
| 11 | `11_Imbalanced_Data.ipynb` | Undersampling, oversampling, SMOTE, Borderline-SMOTE, and class weights compared on the same held-out test set |
| 12 | `12_Data_Splitting.ipynb` | Stratified vs. unstratified splitting (variance demonstrated across 10 trials), fit-on-train discipline shown numerically |
| 13 | `13_Data_Leakage.ipynb` | Every leakage type (target, train-test contamination, feature, preprocessing, temporal) shown as an Incorrect → Correct workflow pair |
| 14 | `14_Preprocessing_Pipeline.ipynb` | A complete `Pipeline` + `ColumnTransformer` encoding every prior notebook's decisions into one reusable object |
| 15 | `15_Before_After_Preprocessing.ipynb` | Full before/after comparison: shape, types, missing values, duplicates, outliers, distributions, categories, ranges, feature count |
| 16 | `16_Complete_Preprocessing_Workflow.ipynb` | The full end-to-end workflow, producing the final `telco_churn_ml_ready.csv` |
| 17 | `17_Mini_Assessment.ipynb` | **Fully answered** self-assessment — every question and coding exercise worked through, not left blank |

## Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab
- Sprint 4 (EDA) — this sprint directly continues its dataset and findings

## Libraries Used

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
imbalanced-learn
statsmodels
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn statsmodels
```

## How to Use This Repository

1. Work through the notebooks in order — each builds on decisions made in earlier ones
   (e.g., Notebook 14's pipeline encodes Notebook 3's imputation choice and Notebook 7's
   encoding choice directly).
2. Run every code cell yourself. Every technique this sprint covers but does **not**
   select for this dataset (mean/median imputation, KNN/Iterative imputation,
   undersampling, label-encoding a nominal column, Box-Cox) is still fully implemented
   and shown — so you can see exactly *why* it was rejected, not just that it was.
3. `16_Complete_Preprocessing_Workflow.ipynb` produces the final deliverable,
   `telco_churn_ml_ready.csv`, plus a fitted `Pipeline` object — the actual hand-off
   point to Sprint 6.
4. Attempt to independently re-derive the answers in `17_Mini_Assessment.ipynb` before
   reading them, as a genuine self-check.

## Key Decisions Carried Into Sprint 6 (Feature Engineering)

- `TotalCharges`'s 11 missing values: filled with **0** (MAR, justified by `tenure == 0`),
  not mean/median.
- 42 near-duplicate customer profiles and 112 multivariate billing outliers:
  **retained**, not removed — the `billing_residual` column is engineered as a candidate
  feature.
- `Contract`: ordinally encoded (0/1/2); all other categoricals: one-hot encoded via
  `OneHotEncoder(handle_unknown='ignore')` for production safety.
- Numeric features: `StandardScaler`, justified by confirmed outlier-freedom.
- Class imbalance: `class_weight='balanced'` selected as the lowest-risk first response;
  SMOTE remains available and fully demonstrated as an alternative.
- All preprocessing is wrapped in a single `sklearn.Pipeline`, structurally preventing the
  data-leakage patterns demonstrated in Notebook 13.

## AI Usage Policy

Learning resources such as ChatGPT, official documentation, Scikit-learn/Pandas/NumPy
docs, books, YouTube, and technical blogs were used to *understand* each technique — not
to copy complete preprocessing pipelines directly. Every implementation, decision, and
piece of reasoning in these notebooks reflects independently-verified work against the
real dataset, re-executed and checked before being finalized.
