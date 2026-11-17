[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1H95nwBjW_EPPLCMjWTuTthRCLTsmBYuo?usp=sharing)

# ML_11 — Classification Evaluation & Performance Metrics

**Stage:** Machine Learning  
**Program:** Data Science & AI  

---

## 1. Overview

A classification model can predict class labels and probabilities, but these outputs do not fully explain how well the model performs.

In this unit, we evaluate the AdventureWorks order-return classifier. The operations team uses the model to identify orders that may be returned and prioritize them for review.

The main goal is not simply to maximize a score. It is to understand the types of mistakes the model makes and choose evaluation metrics that match the business requirements.

## 2. Business Problem

The positive class is `1` (`returned`), and the negative class is `0` (`not returned`).

The review team has limited capacity, and the two main classification errors have different business effects:

| Error | Meaning | Operational effect |
|---|---|---|
| False Positive (FP) | An order is not returned but is flagged for review. | Unnecessary review effort. |
| False Negative (FN) | An order is returned but is not flagged. | A return risk is missed. |

A model that identifies more returned orders may also create more unnecessary reviews. Another model may reduce unnecessary reviews but miss more returns. The best choice depends on the business trade-off.

## 3. Learning Outcomes

By the end of this notebook, learners should be able to:

- Read and interpret a confusion matrix.
- Explain True Positive, True Negative, False Positive, and False Negative.
- Calculate and interpret accuracy, precision, recall, specificity, and F1-score.
- Explain why accuracy can be misleading with imbalanced classes.
- Compare a trained classifier with a simple majority-class baseline.
- Read per-class results using `classification_report`.
- Explain binary, macro, weighted, and micro averaging.
- Use classification metrics to discuss model performance in a business context.

## 4. Notebook Roadmap

| Section | Topic | Main purpose |
|---|---|---|
| 1 | Evaluation starts with the decision | Connect evaluation to business costs and goals. |
| 2 | Confusion matrix | Make correct predictions and classification errors visible. |
| 3 | Core classification metrics | Evaluate different aspects of model performance. |
| 4 | Accuracy under class imbalance | Understand why accuracy alone can be misleading. |
| 5 | Per-class results and averaging strategies | Compare class-level performance and averaging methods. |
| Lab | Inspect class-level performance | Build a per-class metrics table and interpret the results. |

## 5. Dataset and Modeling Setup

The notebook creates a deterministic synthetic dataset representing **2,400 AdventureWorks orders**. It does not require an external CSV file.

The target column is:

- `returned`: `1` means the order was returned; `0` means it was not returned.

The input features include:

**Numerical features**
- `quantity`
- `unit_price`
- `discount_pct`
- `order_value`
- `customer_tenure_months`
- `prior_orders`
- `days_since_last_order`

**Categorical features**
- `product_category`
- `sales_channel`
- `customer_segment`
- `payment_method`
- `region`

The notebook uses a stratified train/test split with 75% of the data for training and 25% for testing. The preprocessing and model workflow includes:

- Median imputation and `StandardScaler` for numerical features.
- Most-frequent imputation and `OneHotEncoder` for categorical features.
- `ColumnTransformer` to apply the correct preprocessing to each feature group.
- `Pipeline` with `LogisticRegression` as the classifier.

The notebook includes a self-contained setup, so it can run independently. It also explains how the model and test set connect to the previous ML_10 workflow.

## 6. Key Concepts

### Confusion Matrix

A confusion matrix compares actual labels with predicted labels.

|  | Actual: Returned | Actual: Not Returned |
|---|---:|---:|
| **Predicted: Returned** | True Positive (TP) | False Positive (FP) |
| **Predicted: Not Returned** | False Negative (FN) | True Negative (TN) |

Always check the positive class and label order before interpreting the matrix.

### Core Metrics

| Metric | Main question | Interpretation |
|---|---|---|
| Accuracy | How many predictions are correct? | Overall proportion of correct predictions. |
| Precision | Of the orders flagged as returned, how many were actually returned? | Important when unnecessary reviews are costly. |
| Recall (Sensitivity) | Of all returned orders, how many did the model identify? | Important when missing a return is costly. |
| Specificity | Of all orders that were not returned, how many were correctly left unflagged? | Measures performance on the negative class. |
| F1-score | What is the balance between precision and recall? | Harmonic mean of precision and recall. |
| Balanced accuracy | How well does the model perform across both classes? | Gives equal importance to recall for each class in binary classification. |

### Accuracy and Class Imbalance

When one class is much more common than the other, a model can achieve high accuracy by mostly or always predicting the majority class.

For example, if 950 out of 1,000 orders were not returned, a model that predicts “not returned” for every order would achieve 95% accuracy while identifying none of the returned orders.

This is why accuracy should be interpreted with other metrics and compared with a baseline.

### Averaging Strategies

Scikit-learn provides several ways to summarize metrics across classes:

| Strategy | How it works | When it is useful |
|---|---|---|
| `binary` | Calculates the metric for the positive class. | Binary classification when the positive class is the main focus. |
| `macro` | Calculates the metric for each class and takes the unweighted mean. | When every class should have equal influence. |
| `weighted` | Calculates the metric for each class and weights it by the number of true examples in that class. | When class frequency should influence the overall result. |
| `micro` | Combines the counts across classes before calculating the metric. | When the overall contribution of individual predictions is important. |

For single-label multiclass classification, micro precision, micro recall, micro F1, and accuracy are equal.

## 7. Important Scikit-learn Functions

| Function | Purpose |
|---|---|
| `confusion_matrix()` | Creates the confusion matrix. |
| `ConfusionMatrixDisplay()` | Visualizes the confusion matrix. |
| `accuracy_score()` | Calculates accuracy. |
| `precision_score()` | Calculates precision. |
| `recall_score()` | Calculates recall. |
| `f1_score()` | Calculates F1-score. |
| `balanced_accuracy_score()` | Calculates balanced accuracy. |
| `classification_report()` | Displays precision, recall, F1-score, and support for each class. |
| `precision_recall_fscore_support()` | Returns precision, recall, F1-score, and support for each class or averaging strategy. |
| `DummyClassifier()` | Creates a simple baseline classifier. |

## 8. Reference Run: An Important Evaluation Lesson

The notebook's saved output provides a useful example of why accuracy must not be used alone.

In the saved run, the test set contains **600 orders**, including **63 returned orders (10.5%)**. The Logistic Regression model predicts every test order as `not returned`:

| Confusion matrix result | Count |
|---|---:|
| True Negative (TN) | 537 |
| False Positive (FP) | 0 |
| False Negative (FN) | 63 |
| True Positive (TP) | 0 |

The model achieves **89.5% accuracy**, but its recall for returned orders is **0.000**. It fails to identify any returned order.

The majority-class `DummyClassifier` produces the same saved baseline scores:

| Metric | Logistic Regression | Majority baseline |
|---|---:|---:|
| Accuracy | 0.895 | 0.895 |
| Precision | 0.000 | 0.000 |
| Recall | 0.000 | 0.000 |
| F1-score | 0.000 | 0.000 |
| Balanced accuracy | 0.500 | 0.500 |

**Teaching point:** A high accuracy score does not automatically mean that a classifier is useful. In this business case, a model that misses every returned order does not meet the operational goal, even though its accuracy appears high.

These values are the saved notebook outputs. Results may differ if the data generation, split, or model configuration is changed.

## 9. Practical Lab — Inspect Class-Level Performance

Complete the final lab in the notebook:

1. Use `precision_recall_fscore_support` with `average=None`.
2. Build a DataFrame with one row for each class.
3. Include precision, recall, F1-score, and support as columns.
4. Identify which class has the lower recall.
5. Compare macro F1 with weighted F1.
6. Explain whether the weighted score gives a more optimistic or pessimistic view than the macro score in this run.

Use class labels `0` and `1` clearly in your output.

## 10. How to Use This Notebook

1. Open the notebook using the **Open in Google Colab** badge at the top.
2. Run the notebook from top to bottom.
3. Review the confusion matrix before studying the metric scores.
4. For every metric, explain what it means for the AdventureWorks review team.
5. Compare Logistic Regression with the majority-class baseline.
6. Complete the practical lab and explain your conclusions.

## 11. Readiness Checklist

Before moving to the next unit, make sure you can:

- [ ] Explain the difference between FP and FN.
- [ ] Interpret all four cells of a confusion matrix.
- [ ] Explain why precision and recall answer different questions.
- [ ] Calculate specificity from TN and FP.
- [ ] Explain why accuracy may be misleading with imbalanced data.
- [ ] Compare a model with a `DummyClassifier` baseline.
- [ ] Explain macro, weighted, and micro averaging.
- [ ] Recommend a metric based on a business requirement.

## 12. Key Takeaway

**Classification evaluation is about understanding the errors a model makes, not just reporting a score.** Choose metrics based on the business objective, class distribution, and cost of false positives and false negatives.
