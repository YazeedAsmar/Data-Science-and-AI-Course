# Notebook 15 — Feature Engineering & Feature Selection

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1H5YUJZs-gdyz0k_B65MLcQ8s6An_YRL8?usp=sharing)

**Program:** Data Science & AI  
**Stage:** Machine Learning  
**Level:** Intermediate  
**Language:** English (B2)  
**Business case:** AdventureWorks — Customer Purchase Prediction

---

## 1. Overview

Machine learning models learn from the input variables provided to them. However, raw data is not always in the best form for learning. Feature engineering helps create more meaningful inputs, while feature selection helps identify a useful subset of those inputs.

In this notebook, you will build a customer-level workflow for predicting whether a customer will make another purchase within the next 30 days.

The notebook covers the full process: understanding the prediction timeline, creating business-focused features, preprocessing data, applying feature selection methods, comparing feature sets, and interpreting feature importance.

## 2. Business Problem

AdventureWorks is an e-commerce business that sells electronics, home products, and accessories. The marketing team wants to identify customers who are likely to place another order within the next 30 days.

Each row represents one customer snapshot.

| Element | Definition |
|---|---|
| Unit of analysis | One customer snapshot |
| Prediction date | October 1, 2026 |
| Observation window | The 90 days before the prediction date |
| Prediction window | The 30 days after the prediction date |
| Target | `repeat_purchase_30d` |
| Business use | Prioritize relevant customer retention campaigns |

**Important:** Features must use information available at the prediction date. Information from the future prediction window must not be used as a model input. This is a key way to prevent data leakage.

## 3. Learning Outcomes

By the end of this notebook, you will be able to:

- Explain the difference between a feature and a target.
- Explain feature engineering and feature selection.
- Create numerical, categorical, date-based, ratio, and interaction features.
- Apply binning and logarithmic transformations.
- Encode categorical variables and scale numerical variables.
- Build preprocessing workflows using `ColumnTransformer` and `Pipeline`.
- Apply filter, wrapper, and embedded feature selection methods.
- Compare models trained with different feature sets.
- Interpret feature importance without confusing association with causation.
- Reduce the risk of data leakage by fitting transformations on training data only.

## 4. Notebook Roadmap

| Section | Topic |
|---|---|
| 1 | Business problem and feature engineering workflow |
| 2 | Generate and inspect customer data |
| 3 | Feature engineering foundations |
| 4 | Build the modeling dataset |
| 5 | Preprocessing with pipelines |
| 6 | Feature selection methods |
| 7 | Evaluate different feature sets |
| 8 | Feature importance and interpretation |
| 9 | Practical exercises |
| 10 | Final applied project |
| 11 | Summary and readiness checklist |

## 5. Dataset

The notebook generates a synthetic customer dataset locally. No external CSV file, download, or API key is required.

- **Number of customers:** 1,200
- **Raw dataset shape:** 1,200 rows × 17 columns
- **Target:** `repeat_purchase_30d`
- **Positive target rate in the saved notebook output:** approximately 32.7%

The generated source data includes customer region and segment, preferred product category, signup and last-order dates, recent order activity, spending, discounts, returns, support tickets, website sessions, and email clicks.

### Main source features

| Feature | Description |
|---|---|
| `orders_90d` | Number of orders in the previous 90 days |
| `days_since_last_order` | Days since the customer's last order |
| `avg_order_value` | Average order value |
| `total_spend_90d` | Total spending in the previous 90 days |
| `items_per_order` | Average items per order |
| `discount_rate` | Customer discount rate |
| `return_rate` | Customer return rate |
| `support_tickets_90d` | Support tickets in the previous 90 days |
| `website_sessions_30d` | Website sessions in the previous 30 days |
| `email_clicks_30d` | Email clicks in the previous 30 days |
| `region` | Customer region |
| `customer_segment` | Customer segment |
| `preferred_category` | Preferred product category |

The notebook also creates `signup_date`, `last_order_date`, and the target variable.

## 6. Feature Engineering

Feature engineering creates or transforms input variables to represent useful business patterns.

### Numerical and behavior features

The notebook creates features such as:

- `spend_per_order`: total spending divided by order count.
- `sessions_per_order`: website sessions relative to order count.
- `engagement_per_session`: email clicks relative to website sessions.

The ratio calculations handle zero denominators to avoid infinite values.

### Date-based features

The notebook derives:

- `customer_tenure_days`
- `last_order_month`
- `last_order_dayofweek`
- `last_order_is_weekend`

In pandas, Monday is day `0` and Sunday is day `6`.

### Other transformations

| Technique | Example in the notebook | Purpose |
|---|---|---|
| Binning | `tenure_group` | Group customers by tenure |
| Log transformation | `log_total_spend_90d` | Reduce the effect of large values |
| Interaction feature | `orders_x_avg_value` | Combine order frequency and average order value |
| Interaction feature | `discount_x_orders` | Combine discount rate and order frequency |
| Reusable function | `create_customer_features()` | Apply feature creation consistently |

A feature should have a clear business meaning and must be available at prediction time. Creating more features does not automatically improve a model.

## 7. Build the Modeling Dataset

The target is stored separately:

- `X` contains the model input features.
- `y` contains `repeat_purchase_30d`.

The notebook excludes the following columns from model inputs:

- `customer_id`: identifier, not a customer behavior feature.
- `repeat_purchase_30d`: target variable.
- `signup_date` and `last_order_date`: raw dates used to derive date-based features.

The saved notebook output shows 1,200 rows and 22 input columns before preprocessing.

### Train-test split

The notebook uses a stratified 80/20 split:

| Split | Rows |
|---|---:|
| Training | 960 |
| Test | 240 |

Stratification keeps the positive target proportions similar in both sets. The saved output shows a positive rate of approximately 32.7% in training and 32.5% in testing.

For real customer snapshots collected over time, a time-based split may be more suitable. The notebook uses a random split for this synthetic, single-snapshot teaching example.

## 8. Preprocessing with Pipelines

The notebook uses a `ColumnTransformer` to apply different transformations to numerical and categorical columns.

| Feature type | Preprocessing |
|---|---|
| Numerical | Median imputation, then `StandardScaler` |
| Categorical | Most-frequent imputation, then `OneHotEncoder` |

The encoder uses `handle_unknown="ignore"` so that unseen categories do not cause transformation errors.

The preprocessor is fitted on the training data only:

- `fit_transform()` is applied to training data.
- `transform()` is applied to test data.

This keeps information from the test set out of the preprocessing fit and helps prevent data leakage.

The saved output contains 32 transformed features before variance filtering.

## 9. Feature Selection Methods

Feature selection chooses a subset of the available input variables. It may reduce complexity, memory use, and training time, and can make a model easier to explain. It does not guarantee better predictive performance.

The notebook explores three main families:

| Family | How it works | Methods in this notebook |
|---|---|---|
| Filter | Scores features using statistical measures | `VarianceThreshold`, Mutual Information, `SelectKBest` |
| Wrapper | Repeatedly fits an estimator and removes features | Recursive Feature Elimination (RFE) |
| Embedded | Selects or ranks features during model fitting | L1 Logistic Regression, Random Forest importance |

### Filter methods

- **VarianceThreshold:** removes features with variance below a selected threshold. In the saved output, all 32 transformed features remain.
- **Mutual Information:** estimates how much information a feature provides about the target. Higher scores indicate more estimated dependence, not causation.
- **SelectKBest:** keeps the top K features according to a scoring function. The notebook uses mutual information and selects up to 15 features.

### Wrapper method — RFE

Recursive Feature Elimination repeatedly fits an estimator and removes the least important features. The notebook uses a `RandomForestClassifier` and selects 12 features.

RFE can require more computation because it fits the estimator multiple times.

### Embedded methods

- **L1 Logistic Regression:** L1 regularization can shrink some coefficients to zero. The notebook uses a scaled pipeline and `C=0.15`.
- **Random Forest feature importance:** ranks features using impurity reduction across the fitted trees.

Tree-based importance has limitations. It can favor features with many possible split points, and correlated features can affect each other's importance. Importance does not prove that a feature causes the target.

## 10. Compare Feature Sets

The notebook evaluates the same Random Forest classifier using three feature sets:

1. All features after preprocessing — 32 features.
2. Mutual Information top K — 15 features.
3. RFE selected features — 12 features.

The saved comparison reports the following test results:

| Feature set | Number of features | F1-score | ROC-AUC |
|---|---:|---:|---:|
| All features after preprocessing | 32 | 0.481 | 0.704 |
| Mutual Information top K | 15 | 0.475 | 0.700 |
| RFE selected features | 12 | 0.470 | 0.704 |

**Interpretation:** In this saved run, RFE uses 12 features and reaches the same ROC-AUC as the 32-feature set, while the all-feature model has a slightly higher F1-score. The results show why feature selection should be evaluated rather than assumed to improve performance.

The notebook also discusses practical considerations such as stability across validation folds, interpretability, and the cost of collecting and maintaining features.

## 11. Feature Importance

The notebook introduces two approaches:

- **Tree-based feature importance:** estimates importance from impurity reduction in a fitted Random Forest.
- **Permutation importance:** measures the decrease in model score after a feature is randomly shuffled.

Permutation importance is calculated using the test set for the final explanatory analysis, after the feature-set comparison. It uses ROC-AUC scoring and eight repeats.

A positive permutation importance means the model's ROC-AUC decreased after shuffling that feature. Values near zero suggest limited reliance in this evaluation. Negative values can occur because of sampling variation or noise.

When features are correlated, shuffling one may have a small effect because another feature provides similar information. Do not interpret importance as causality.

## 12. Practical Exercises

The notebook includes five learning activities:

1. **Create two business features:** `support_tickets_per_order` and `clicks_per_session`, handling zero denominators safely.
2. **Inspect preprocessing output:** count numerical and categorical inputs and explain one-hot encoding and train-only fitting.
3. **Compare selection strategies:** compare Mutual Information, RFE, L1, and Random Forest rankings.
4. **Engineer a customer loyalty feature:** create `loyalty_value_score` from orders, average order value, and customer tenure.
5. **Compare different values of K:** test at least three `SelectKBest` values and compare ROC-AUC and feature counts.

Student workspaces are provided, along with instructor solutions for the exercises.

## 13. Final Applied Project — AdventureWorks Customer Retention Features

### Business brief

Build a reproducible feature engineering and feature selection workflow to identify customers likely to make another purchase within the next 30 days.

### Project requirements

1. **Understand the data:** identify the unit of analysis, observation window, prediction date, target, identifiers, and possible leakage fields.
2. **Engineer features:** create at least four features, including a ratio, a date-based feature, a customer behavior feature, and an interaction or transformed feature.
3. **Prepare the data:** split training and test data and build numerical and categorical preprocessing pipelines.
4. **Select features:** apply at least two selection methods and explain why their results may differ.
5. **Train and evaluate:** compare all-feature and selected-feature workflows using ROC-AUC, precision, recall, and F1-score.
6. **Communicate findings:** prepare a short business summary with useful features, method, model performance, limitations, and a recommendation.

### Deliverables

- Completed notebook.
- Feature dictionary describing each engineered feature.
- Model comparison table.
- Short business recommendation.

### Evaluation rubric

| Assessment area | Weight |
|---|---:|
| Data understanding and leakage awareness | 15% |
| Quality and relevance of engineered features | 25% |
| Correct preprocessing workflow | 20% |
| Feature selection and comparison | 20% |
| Model evaluation and interpretation | 10% |
| Clarity of business recommendations | 10% |
| **Total** | **100%** |

## 14. Readiness Checklist

Before moving to the next unit, confirm that you can:

- [ ] Explain the difference between a feature and a target.
- [ ] Explain feature engineering and feature selection.
- [ ] Create ratio, date-based, binned, and interaction features.
- [ ] Apply one-hot encoding and numerical scaling.
- [ ] Build a `ColumnTransformer` and `Pipeline`.
- [ ] Explain why preprocessing must be fitted on training data only.
- [ ] Apply Mutual Information, RFE, and L1-based selection.
- [ ] Compare models trained on different feature sets.
- [ ] Interpret permutation importance carefully.
- [ ] Identify and prevent feature leakage.

## 15. Key Takeaways

1. Feature quality and relevance matter as much as the choice of algorithm.
2. Feature engineering should be guided by the business problem and prediction time.
3. A feature must be available when the prediction is made.
4. Pipelines help make preprocessing repeatable and reduce data leakage.
5. Filter, wrapper, and embedded methods use different definitions of feature usefulness.
6. Feature selection does not guarantee better performance.
7. Feature importance is useful for interpretation, but it does not prove causality.
8. Keep the test set separate from feature selection and model development.

**Recommended next step:** Continue to Unsupervised Learning & Clustering, where you will discover customer groups without using a target variable.

## References

- [Scikit-learn — Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Scikit-learn — ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Scikit-learn — Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Scikit-learn — Pipeline and composite estimators](https://scikit-learn.org/stable/modules/compose.html)
- [Scikit-learn — Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [Pandas — Time series and date functionality](https://pandas.pydata.org/docs/user_guide/timeseries.html)

---

**End of Notebook 15 — Feature Engineering & Feature Selection**
