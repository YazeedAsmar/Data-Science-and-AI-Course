[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TF7l6nYu015KTtlRrRcIRKmnpdRr8F6w?usp=sharing)

# Unit 10 — Logistic Regression: Classification Fundamentals

**Stage 2: Machine Learning**  
**Level:** Beginner  
**Language:** English (B2)  
**Business scenario:** AdventureWorks — Order Return Risk  
**Environment:** Google Colab / Jupyter Notebook

---

## Overview

In this unit, learners move from regression to **binary classification**. The business scenario is based on AdventureWorks, where the operations team wants to identify newly placed orders that may be returned.

The notebook uses Logistic Regression to estimate the probability that an order will be returned. Learners explore how a linear score is converted into a probability, prepare numerical and categorical features, train a classification pipeline, inspect predictions, and select a decision threshold based on an operational review-capacity limit.

The central idea is that a model's probability is not automatically a business decision. The organization must choose how to act on that probability.

## Business Question

Which newly placed orders should the operations team prioritize for review when it cannot manually review every order?

**Target:** `returned`

- `1` means the order was returned.
- `0` means the order was not returned.

**Prediction moment:** Immediately after an order is placed. Only information available at that moment may be used.

### Prediction Features

The notebook uses the following features:

- `order_value`
- `item_count`
- `discount_rate`
- `product_category`
- `sales_channel`
- `customer_tenure_months`
- `payment_method`

**Excluded fields:**

- `order_id` is an identifier, not a predictive feature.
- `refund_amount` and `return_reason` are only available after the return outcome.
- `returned` is the target and must not be included in the input features.

Using information that becomes available only after the outcome would create data leakage and would not reflect the real prediction scenario.

## Learning Outcomes

By the end of this unit, learners will be able to:

- Frame a business question as a binary classification problem.
- Explain how Logistic Regression converts a linear score into a probability.
- Describe the role of the sigmoid function.
- Distinguish predicted probabilities from predicted class labels.
- Explain the default 0.50 decision threshold and why it is not always the best business choice.
- Prepare mixed numerical and categorical features for Logistic Regression.
- Build a preprocessing and classification workflow using `ColumnTransformer` and `Pipeline`.
- Train a Logistic Regression classifier and inspect its learned coefficients.
- Use `predict_proba()` and `predict()` correctly.
- Compare thresholds using precision, recall, F1-score, and confusion-matrix counts.
- Select a threshold that respects a business review-capacity constraint.

## Notebook Roadmap

| Section | Topic | What You Will Learn |
|---|---|---|
| 1 | The Classification Decision | Define the business decision, target, prediction moment, and valid input features. |
| 2 | From a Linear Score to a Probability | Understand Logistic Regression and the sigmoid function. |
| 2.1 | Read the Model Output | Distinguish a probability estimate from a final class label. |
| 3 | Build the AdventureWorks Workflow | Generate and inspect the order dataset and prepare the feature matrix and target. |
| 3.1 | Prepare Mixed Data Without Leakage | Impute missing values, scale numerical features, encode categories, and split the data. |
| 4 | Interpret Predictions and Coefficients | Inspect class order, predicted probabilities, predicted labels, and example predictions. |
| 5 | Make a Threshold-Based Business Decision | Compare candidate thresholds and their operational impact. |
| 5.1 | Translate Results into an Operating Choice | Select a feasible threshold under the review-capacity limit. |
| 6 | Applied Mini-Project | Use the workflow and evidence to recommend a practical classification decision. |

## Key Concepts

### 1. Regression vs. Classification

Regression estimates a numerical quantity, such as order revenue. Classification estimates a category, such as whether an order will be returned.

This notebook focuses on **binary classification**, because the target has two possible values: returned (`1`) or not returned (`0`).

The model does not determine with certainty whether an order will be returned. It estimates risk from patterns in historical examples. The operations team decides what action to take.

### 2. Logistic Regression and the Sigmoid Function

Despite the word *regression* in its name, Logistic Regression is used here as a classification algorithm.

The model first calculates a linear score from the input features and their learned coefficients. It then applies the sigmoid function to map the score to a value between 0 and 1.

\[
p = \frac{1}{1 + e^{-z}}
\]

Where:

- \(z\) is the linear score.
- \(p\) is the estimated probability of the positive class.

The sigmoid function converts any real-valued score into a probability-like output between 0 and 1.

### 3. Probability vs. Predicted Class

A predicted probability is not the same as a final class decision.

For example, a return probability of `0.72` means the model estimates a 72% probability for the positive class. Whether the order is flagged depends on the selected threshold.

The notebook demonstrates the default threshold of **0.50**:

- Probability greater than or equal to 0.50 → predict class `1`.
- Probability below 0.50 → predict class `0`.

The 0.50 threshold is a common default, not a universal business rule. A business may choose another threshold based on its costs, risks, and operational capacity.

### 4. Prepare Numerical and Categorical Features

Logistic Regression requires numerical input and can be sensitive to feature scale. The notebook applies separate preprocessing steps to each feature group.

| Feature Group | Preprocessing |
|---|---|
| Numerical | Fill missing values using the median, then standardize with `StandardScaler`. |
| Categorical | Fill missing values using the most frequent category, then apply `OneHotEncoder(handle_unknown="ignore")`. |

The notebook uses `ColumnTransformer` to apply the correct transformations to each group of columns.

### 5. Train and Evaluate Without Data Leakage

The notebook creates three data partitions:

| Partition | Share of Full Dataset | Purpose |
|---|---:|---|
| Training | 60% | Fit preprocessing and train the classifier. |
| Validation | 20% | Compare thresholds and make the operating choice. |
| Test | 20% | Reserved for final evaluation after decisions are fixed. |

The splits use `stratify` to preserve the target-class proportions across the partitions.

All preprocessing and the classifier are combined in a `Pipeline`. The pipeline is fitted on the training data, so the validation and test data do not influence the imputation values, scaling parameters, or category mappings.

### 6. Understand the Model Outputs

The notebook uses two important methods:

- `predict_proba()` returns estimated probabilities for each class.
- `predict()` returns class labels using the estimator's default decision rule.

The probability columns follow the order in `classifier.classes_`. The notebook checks this class order before selecting the probability for class `1`.

It also creates a prediction review table containing the actual class, predicted class, estimated return probability, and whether the prediction is correct.

### 7. Decision Thresholds and Operational Trade-offs

A threshold determines when an order is flagged for review.

- A **lower threshold** generally flags more orders. It may capture more actual returns, but it can also increase unnecessary reviews.
- A **higher threshold** generally flags fewer orders. It can reduce the review queue, but may miss more actual returns.

The notebook compares these candidate thresholds:

`0.25`, `0.35`, `0.50`, `0.65`, and `0.75`.

For each threshold, it reports:

- Number of flagged orders.
- Precision.
- Recall.
- F1-score.
- Missed returns.
- Unnecessary reviews.

### 8. Select a Threshold Based on Review Capacity

The business scenario assumes that the operations team can review at most **15% of incoming orders**.

The notebook calculates the maximum number of orders that can be reviewed, filters out thresholds that exceed this capacity, and sorts feasible options by recall and then precision.

This is a constrained business decision. The threshold with the highest F1-score is not automatically the best choice if it exceeds the team's review capacity.

**Important:** Threshold selection is performed using validation data. The test set should remain untouched until the threshold and operating decision have been fixed.

## Dataset

The notebook creates a synthetic AdventureWorks-style order dataset in code. It generates **2,400 orders** with numerical and categorical features, then creates a binary `returned` target using a simulated return-risk pattern with random variation.

A small number of missing values are added to selected features:

- `order_value`
- `customer_tenure_months`
- `product_category`

**No external CSV file is required** for this notebook. The dataset is generated when the data-generation cell runs.

The generated data is designed for learning the classification workflow. It is not a real operational record of customer returns.

## Tools and Libraries

The notebook uses:

- **NumPy** — numerical operations and synthetic data generation.
- **Pandas** — create and inspect the order dataset and prediction reports.
- **Matplotlib** — visualize the sigmoid function.
- **scikit-learn**
  - `ColumnTransformer`
  - `SimpleImputer`
  - `StandardScaler`
  - `OneHotEncoder`
  - `Pipeline`
  - `LogisticRegression`
  - `train_test_split`
  - `precision_score`
  - `recall_score`
  - `f1_score`
  - `confusion_matrix`

## How to Use This Notebook

1. Open the notebook using the **Open in Google Colab** button at the top of this page.
2. Run the notebook cells in order.
3. Review the business question and identify which features are available at prediction time.
4. Explore how the sigmoid function maps a linear score to a probability.
5. Run the data-generation and preprocessing steps.
6. Review the model's predicted probabilities and class labels.
7. Compare the candidate thresholds using the validation results.
8. Identify which thresholds satisfy the 15% review-capacity limit.
9. Explain the trade-off between capturing more returns and keeping the review queue manageable.

## Practical Challenge

After completing the notebook:

1. Change the candidate threshold list and compare the new results.
2. Explain how lowering the threshold affects the number of flagged orders.
3. Compare precision and recall. Why can improving one affect the other?
4. Identify a threshold that satisfies the 15% review-capacity constraint.
5. Explain why threshold selection should use validation data rather than the test set.
6. Identify one feature that would cause data leakage if it were used at the prediction moment.
7. Explain why the model's probability should support, rather than replace, the business decision.

## Readiness Checklist

- [ ] I can explain why this is a binary classification problem.
- [ ] I can describe how Logistic Regression uses the sigmoid function.
- [ ] I can distinguish predicted probabilities from predicted labels.
- [ ] I understand the default 0.50 threshold.
- [ ] I can explain the numerical and categorical preprocessing steps.
- [ ] I understand why preprocessing is fitted on training data only.
- [ ] I can explain precision, recall, and F1-score.
- [ ] I can describe the trade-off between lower and higher thresholds.
- [ ] I can select a threshold under a review-capacity constraint.
- [ ] I understand why the test set should remain untouched during threshold selection.

## Key Takeaway

**Logistic Regression estimates the probability of a class, but the decision threshold connects that probability to a real business action. A useful classification solution must consider model performance and operational constraints together.**

## References

- [scikit-learn — Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [scikit-learn — Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [scikit-learn — Pipeline and Composite Estimators](https://scikit-learn.org/stable/modules/compose.html)
- [scikit-learn — OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
