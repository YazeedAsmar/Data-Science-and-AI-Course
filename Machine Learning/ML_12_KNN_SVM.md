[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RHequHbQiT1sM2C7RHRTaNBQ8WBK7n4B?usp=sharing)

# ML_12 — K-Nearest Neighbors (KNN) & Support Vector Machines (SVM)

**Program:** Data Science & Artificial Intelligence  
**Stage:** Machine Learning  
**Level:** Applied Machine Learning  
**Language:** English (B2)  
**Business Case:** AdventureWorks — E-commerce Return Risk

---

## 1. Overview

In this unit, learners explore two supervised machine learning algorithms for classification: **K-Nearest Neighbors (KNN)** and **Support Vector Machines (SVM)**.

The business scenario is based on AdventureWorks. The Operations team wants to identify orders that have a high risk of being returned before dispatch. The team has limited capacity to review orders, so the model should identify useful risk patterns while keeping unnecessary reviews under control.

The notebook uses a reproducible synthetic dataset created in Python. It represents a teaching simulation, not real customer records or actual company transactions.

## 2. Business Problem

The target column is `return_risk`:

- `1`: the order is classified as high risk of being returned.
- `0`: the order is not classified as high risk.

This is a **binary classification** problem.

The model uses information assumed to be available when an order is prepared for dispatch. Information that becomes available only after a return decision must not be used as an input, because it could cause target leakage.

### Business trade-off

- **False Negative:** a high-risk order is not identified. The team may miss an opportunity to take action.
- **False Positive:** a standard-risk order is flagged. The team spends time reviewing an order that may not need attention.

The best model depends on the operational cost of these two errors, not only on accuracy.

## 3. Learning Outcomes

By the end of this notebook, learners should be able to:

- Explain how KNN classifies a new observation using nearby training examples.
- Calculate and compare Euclidean and Manhattan distances.
- Explain why feature scaling is important for distance-based algorithms.
- Build a KNN classifier using `Pipeline` and `StandardScaler`.
- Explain the effects of `k` and neighbor weighting.
- Describe SVM decision boundaries, margins, and support vectors.
- Explain the roles of `C`, `gamma`, and kernel selection.
- Compare linear, RBF, and polynomial SVM kernels.
- Tune model parameters using stratified cross-validation and `GridSearchCV`.
- Compare KNN and SVM using a separate test set and business-focused metrics.

## 4. Notebook Roadmap

| Part | Sections | Main topics |
|---|---|---|
| Data preparation | 1–4 | Environment, business problem, synthetic dataset, feature/target split |
| Part A — KNN | 5–11 | KNN intuition, distance measures, scaling, `k`, weighting, strengths and limitations |
| Part B — SVM | 12–20 | Margins, support vectors, `C`, kernels, `gamma`, decision scores, cross-validation tuning |
| Part C — Model Evaluation | 21–26 | Model selection, test evaluation, confusion matrices, ROC curves, common mistakes |
| Part D — Applied Mini Project | 27–31 | Return-risk model selection report, knowledge check, key takeaways |

## 5. Dataset

The notebook generates a deterministic synthetic dataset containing **1,800 order records** and **8 columns**. A fixed random seed (`RANDOM_STATE = 42`) makes the data generation reproducible.

### Input features

| Feature | Description |
|---|---|
| `order_value` | Total value of the order |
| `items_count` | Number of items in the order |
| `customer_orders` | Number of previous completed customer orders |
| `discount_pct` | Discount applied to the order |
| `delivery_delay_days` | Expected delivery delay relative to the promise |
| `days_since_last_order` | Days since the customer's previous order |
| `product_risk_score` | Historical category-level return tendency, represented as a score from 0 to 1 |

### Target

| Column | Meaning |
|---|---|
| `return_risk` | `1` = high return risk; `0` = standard risk |

In the saved notebook run, 58.6% of records belong to the high-risk class. The data is generated for learning and should not be treated as a real business benchmark.

**No external CSV file is required.** The notebook creates the dataset directly in its code cells.

## 6. Part A — K-Nearest Neighbors (KNN)

### How KNN works

KNN is an instance-based algorithm. It stores the training examples and uses them when a prediction is requested.

For classification:

1. Choose the number of neighbors, `k`.
2. Calculate the distance from the new observation to the training observations.
3. Find the `k` closest observations.
4. Use their class labels to vote.
5. Assign the majority class.

### Distance measures

| Distance | Description |
|---|---|
| Euclidean (L2) | Straight-line distance between two points |
| Manhattan (L1) | Sum of the absolute differences between feature values |
| Minkowski | A general distance family that includes Euclidean and Manhattan distance |

### Why feature scaling matters

KNN relies on distance. If features have very different numerical ranges, a feature with large values can dominate the distance calculation.

The notebook uses `StandardScaler`:

\[
z = \frac{x-\mu}{\sigma}
\]

The scaler is placed inside a `Pipeline`. This ensures that it is fitted only on the training portion of each cross-validation split.

### Important KNN parameters

| Parameter | Purpose |
|---|---|
| `n_neighbors` | Number of nearest observations (`k`) used for voting |
| `weights="uniform"` | Gives every neighbor the same voting weight |
| `weights="distance"` | Gives closer neighbors more influence |
| `metric` / `p` | Controls the distance calculation; Minkowski with `p=2` is Euclidean, while `p=1` is Manhattan |

The notebook compares multiple values of `k`, studies uniform and distance weighting, and tunes KNN with cross-validation.

## 7. Part B — Support Vector Machines (SVM)

### Decision boundary, margin, and support vectors

SVM is a supervised learning algorithm used for classification and regression. For classification, it learns a decision boundary that separates classes.

A linear decision boundary can be represented as:

\[
w^T x + b = 0
\]

The **margin** is the distance between the decision boundary and the closest training observations. The observations that define the margin are called **support vectors**.

SVM aims to find a boundary with a wide margin while controlling classification errors.

### The `C` parameter

`C` controls the trade-off between margin width and training errors.

| Setting | General effect |
|---|---|
| Smaller `C` | Stronger regularization; allows more training violations and favors a wider, smoother margin |
| Larger `C` | Weaker regularization; places more emphasis on correctly classifying training observations and may create a more complex boundary |

### SVM kernels

| Kernel | Description |
|---|---|
| `linear` | Uses a linear decision boundary |
| `rbf` | Can represent flexible non-linear decision boundaries |
| `poly` | Represents polynomial decision boundaries |

### The RBF `gamma` parameter

`gamma` controls how local the influence of each training observation is:

- **Low gamma:** each observation has broader influence, often producing a smoother boundary.
- **High gamma:** each observation has more local influence, which can create a highly detailed boundary and may fit noise.

The notebook compares kernel choices and uses `GridSearchCV` to tune `C` and `gamma` for an RBF SVM.

### SVM decision scores

By default, `SVC` returns class predictions and decision scores. The `decision_function` score is useful for ranking observations and calculating ROC-AUC, but it is **not a calibrated probability**.

If a workflow needs probability estimates, a suitable probability-calibration method is required.

## 8. Model Selection and Evaluation

The notebook follows a leakage-aware workflow:

1. Generate the dataset.
2. Separate features and target.
3. Create a stratified 80/20 train/test split.
4. Use stratified cross-validation on the training data for model selection.
5. Keep preprocessing inside pipelines.
6. Evaluate the selected models on the held-out test set.

The training set contains 1,440 observations, and the test set contains 360 observations.

### Main evaluation metrics

| Metric | What it tells us |
|---|---|
| Accuracy | Proportion of all predictions that are correct |
| Precision | Of the orders flagged as high risk, how many are actually high risk |
| Recall | Of all high-risk orders, how many the model identifies |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | How well the model ranks positive cases above negative cases across thresholds |

The notebook also uses confusion matrices to inspect false positives and false negatives, and ROC curves to compare ranking performance across thresholds.

## 9. Reference Results from the Saved Notebook

The following values are the outputs saved in the uploaded notebook. They are specific to its synthetic dataset, split, and parameter search; they are not real-world performance guarantees.

### KNN: scaling comparison

| Model | Test accuracy |
|---|---:|
| KNN without scaling | 0.542 |
| KNN with `StandardScaler` | 0.642 |

This demonstrates how scaling can change distance calculations and model performance.

### Cross-validation highlights

| Experiment | Saved result |
|---|---|
| KNN `k` comparison | Highest listed mean F1: 0.720 at `k = 29` |
| KNN weighting | Uniform: F1 0.710; distance: F1 0.712 |
| Linear SVM, `C` comparison | Highest listed mean F1: 0.744 at `C = 0.01` |
| RBF SVM grid search | Best `C = 0.1`, `gamma = "scale"`; mean CV F1 = 0.752 |
| KNN grid search | Best `k = 15`, `p = 1`, `weights = "uniform"`; mean CV F1 = 0.723 |

### Final test-set comparison

| Metric | KNN | RBF SVM |
|---|---:|---:|
| Accuracy | 0.650 | 0.658 |
| Precision | 0.679 | 0.656 |
| Recall | 0.763 | 0.877 |
| F1-score | 0.719 | 0.751 |
| ROC-AUC | 0.686 | 0.696 |

**Interpretation:** In this saved run, RBF SVM has higher recall and F1-score, while KNN has higher precision. The SVM identifies a larger share of high-risk orders, but it also has lower precision. The preferred model depends on the team's review capacity and the relative cost of missed high-risk orders versus unnecessary reviews.

## 10. Applied Mini Project — AdventureWorks Return-Risk Model Selection

The Operations Manager wants a recommendation for a model that can flag high-risk orders before dispatch.

### Tasks

1. Inspect the dataset and describe the target distribution.
2. Split the data into training and test sets using stratification.
3. Build a scaled KNN pipeline.
4. Compare at least five values of `k`.
5. Build a linear SVM and an RBF SVM.
6. Tune the RBF SVM using cross-validation.
7. Compare the selected KNN and SVM on the held-out test set.
8. Review confusion matrices, recall, precision, F1-score, and ROC-AUC.
9. Recommend one model and explain the trade-offs.
10. State two limitations of the experiment and one next step before deployment.

### Deliverable

Prepare a short model-selection report containing:

- Business objective
- Data and validation approach
- Models and parameter settings
- Evaluation results
- Error analysis
- Final recommendation
- Limitations and next steps

## 11. Common Mistakes to Avoid

- Skipping feature scaling for KNN or SVM.
- Scaling the complete dataset before splitting it into training and test sets.
- Tuning model parameters using the test set.
- Using accuracy alone to select a classifier.
- Assuming that a larger `k` always improves KNN.
- Assuming that a more complex SVM kernel is always better.
- Treating SVM decision scores as calibrated probabilities.
- Ignoring prediction cost as the dataset grows.

## 12. How to Use This Notebook

1. Open the notebook using the **Open in Google Colab** badge at the top.
2. Run the notebook from top to bottom.
3. Review the generated dataset and target distribution.
4. Follow Part A to understand and tune KNN.
5. Follow Part B to understand and tune SVM.
6. Compare the selected models in Part C.
7. Complete the applied mini project and knowledge check.
8. Explain your final model recommendation using the business context.

## 13. Readiness Checklist

Before moving to the next topic, make sure you can:

- [ ] Explain how KNN assigns a class to a new observation.
- [ ] Explain Euclidean and Manhattan distance.
- [ ] Explain why feature scaling matters.
- [ ] Describe the purpose of `k` and neighbor weighting.
- [ ] Explain a decision boundary, margin, and support vectors.
- [ ] Describe the effects of `C` and `gamma`.
- [ ] Compare linear, RBF, and polynomial kernels.
- [ ] Explain why SVM decision scores are not probabilities.
- [ ] Use cross-validation for model selection.
- [ ] Evaluate models using a separate test set.
- [ ] Recommend a model based on business costs and operational constraints.

## 14. Key Takeaway

**KNN predicts from nearby examples, while SVM learns a decision boundary by balancing margin size and classification errors.** Both algorithms usually need feature scaling, and both should be evaluated using a leakage-safe validation process. The best model is the one that meets the business objective and operational constraints—not necessarily the one with the highest value for a single metric.

## 15. Further Reading

- [scikit-learn — Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [scikit-learn — KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [scikit-learn — Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [scikit-learn — SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [scikit-learn — Importance of Feature Scaling](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html)
