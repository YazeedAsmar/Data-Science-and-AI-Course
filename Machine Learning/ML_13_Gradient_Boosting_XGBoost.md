[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dzQC2gffpTwdLL7ubvIYvAx1RiXXsztl?usp=sharing)

# ML_13 — Gradient Boosting & XGBoost

**Program:** Data Science & Artificial Intelligence  
**Stage:** Machine Learning  
**Level:** Applied Machine Learning  
**Language:** English (B2)  
**Business Case:** AdventureWorks — Return Risk Intelligence  
**Prerequisite:** ML_10–ML_12 — Classification Fundamentals and Model Evaluation

---

## 1. Overview

This unit introduces **Gradient Boosting** and **XGBoost**, two important ensemble-learning approaches for classification.

The notebook uses an AdventureWorks e-commerce scenario. The Operations team wants to identify orders with a high risk of being returned so that it can prioritize manual checks before fulfillment.

Learners first understand the general idea of ensemble learning and the difference between bagging and boosting. They then train a scikit-learn Gradient Boosting classifier, explore its parameters, train an XGBoost model, compare both approaches using cross-validation, and evaluate a tuned model.

The notebook also covers feature importance, threshold selection, model limitations, and business recommendations.

## 2. Business Problem

The target column is `return_risk`:

- `1`: high return risk.
- `0`: normal return risk.

The model should use information available when an order is placed. Information that becomes available only after delivery or after a return has occurred must not be used as an input because it can cause **target leakage**.

The two main operational errors are:

| Error | Meaning | Possible business impact |
|---|---|---|
| False Positive (FP) | A normal-risk order is flagged as high risk. | Unnecessary manual review. |
| False Negative (FN) | A high-risk order is not flagged. | A potentially risky order is missed. |

The final model and classification threshold should reflect review capacity and the relative costs of these errors.

## 3. Learning Outcomes

By the end of this notebook, learners should be able to:

- Explain the difference between bagging and boosting.
- Describe how Gradient Boosting builds an ensemble sequentially.
- Explain the roles of `learning_rate`, `n_estimators`, and tree complexity.
- Train and evaluate a `GradientBoostingClassifier`.
- Explain important XGBoost parameters and regularization controls.
- Train an `XGBClassifier` for binary classification.
- Compare models using stratified cross-validation.
- Tune XGBoost with `GridSearchCV`.
- Evaluate a selected model using classification metrics and diagnostic curves.
- Interpret feature importance without making causal claims.
- Explain how classification thresholds affect precision, recall, and review workload.
- Present a model recommendation in business terms.

## 4. Notebook Roadmap

| Section | Topic | Main focus |
|---|---|---|
| 1 | Environment setup | Libraries and reproducibility |
| 2 | Business problem and dataset | Target, features, and leakage awareness |
| 3 | Ensemble learning | Bagging versus boosting |
| 4 | Gradient Boosting intuition | Sequential learning and loss reduction |
| 5 | Gradient Boosting classifier | Baseline model and parameter experiments |
| 6 | Boosting parameters | Learning rate, estimators, tree depth, and related controls |
| 7 | XGBoost | Concepts, parameters, and baseline implementation |
| 8 | Model comparison | Stratified cross-validation |
| 9 | Hyperparameter tuning | `GridSearchCV` |
| 10 | Final evaluation | Classification report and ROC-AUC |
| 11 | Model interpretation | Feature and permutation importance |
| 12 | Threshold selection | Precision, recall, and flagged-order volume |
| 13 | Mini project and knowledge check | Business recommendation and readiness review |

## 5. Dataset

The notebook generates a reproducible synthetic dataset directly in Python. It contains **2,400 orders** and does not require an external CSV file.

The target is generated from a transparent synthetic risk process. The dataset is designed for learning and should not be treated as real company data or a production benchmark.

### Input features

| Feature | Description |
|---|---|
| `order_value` | Total order value in currency units |
| `items_count` | Number of items in the order |
| `discount_rate` | Discount applied to the order |
| `delivery_days` | Estimated delivery time at checkout |
| `customer_orders` | Number of previous customer orders |
| `customer_return_rate` | Historical return rate before the current order |
| `is_express_shipping` | Whether express shipping was selected |
| `category_encoded` | Numeric category code |

### Target

| Column | Meaning |
|---|---|
| `return_risk` | `1` = high risk; `0` = normal risk |

In the saved notebook output, the high-risk rate is **29.8%**. The dataset contains no missing values in the saved inspection output.

### Data split

The notebook uses:

- `RANDOM_STATE = 42`
- Stratified train/test split
- 80% training data: 1,920 rows
- 20% test data: 480 rows

The test set is separated before model development and is intended for final evaluation.

## 6. Ensemble Learning: Bagging vs. Boosting

An ensemble combines multiple models to produce a prediction.

| Aspect | Bagging | Boosting |
|---|---|---|
| Training process | Models are trained largely independently | Learners are added sequentially |
| Main idea | Aggregate models to reduce variance | Improve the ensemble stage by stage |
| Common example | Random Forest | Gradient Boosting and XGBoost |
| Main consideration | Diversity between learners | Learning rate, number of stages, and learner complexity |

Boosting trains learners sequentially because each new learner is added to improve the current ensemble. It does not simply replace the previous learner.

Boosting is not automatically better than bagging. The result depends on the data, validation strategy, parameter choices, and business objective.

## 7. Gradient Boosting

### How it works

Gradient Boosting builds an additive model through a sequence of stages:

1. Start with an initial prediction.
2. Measure the current model's error using a loss function.
3. Fit a small decision tree to information derived from the negative gradient of the loss.
4. Add the new tree's contribution to the ensemble.
5. Repeat for the selected number of boosting stages.

For classification, scikit-learn's `GradientBoostingClassifier` commonly optimizes log loss. The individual trees contribute to the ensemble's prediction scores.

### Main parameters

| Parameter | Meaning | General effect |
|---|---|---|
| `n_estimators` | Number of boosting stages | More stages increase model capacity and training time |
| `learning_rate` | Contribution of each stage | Smaller values make updates more conservative |
| `max_depth` | Maximum depth of each tree | Deeper trees can capture more complex patterns |
| `min_samples_leaf` | Minimum samples required at a leaf | Larger values can produce smoother trees |
| `subsample` | Fraction of training rows used per stage | Values below 1 introduce stochasticity |
| `max_features` | Number or fraction of features considered | Can reduce correlation and model complexity |

The learning rate and number of estimators should be considered together. A smaller learning rate often requires more trees, but there is no universally best combination.

### Notebook baseline

The initial `GradientBoostingClassifier` uses:

- `n_estimators=120`
- `learning_rate=0.05`
- `max_depth=2`
- `min_samples_leaf=12`

The saved test output is:

| Metric | Gradient Boosting baseline |
|---|---:|
| Accuracy | 0.698 |
| Precision | 0.464 |
| Recall | 0.091 |
| F1-score | 0.152 |
| ROC-AUC | 0.630 |

The low recall means that the baseline identifies only a small share of high-risk orders at the default prediction threshold.

## 8. XGBoost

**XGBoost** is an optimized implementation of gradient-boosted decision trees. It combines the boosting learning approach with regularization and computational optimizations.

Important ideas include:

- Sequential additive boosting.
- Regularization of tree complexity and leaf weights.
- Gradient and second-order information to guide optimization.
- Efficient computation and tree construction.
- Learned default directions for missing values in tree splits.

### Main XGBoost parameters

| Parameter | Purpose |
|---|---|
| `n_estimators` | Maximum number of boosting rounds |
| `learning_rate` (`eta`) | Shrinks the contribution of each boosting step |
| `max_depth` | Maximum depth of each tree |
| `min_child_weight` | Minimum sum of instance weight required in a child |
| `subsample` | Fraction of rows sampled per boosting round |
| `colsample_bytree` | Fraction of features sampled per tree |
| `gamma` | Minimum loss reduction required to make a split |
| `reg_lambda` | L2 regularization on leaf weights |
| `reg_alpha` | L1 regularization on leaf weights |
| `scale_pos_weight` | Class-weighting control often considered for imbalanced data |

### Binary classification objective

The notebook uses:

`objective="binary:logistic"`

This objective is designed for binary classification and produces probability estimates between 0 and 1. The `predict()` method converts these probabilities into class labels using the default threshold.

A probability is not a business decision by itself. The threshold should reflect false-positive and false-negative costs and the available review capacity.

### XGBoost baseline

The notebook's baseline uses 250 estimators, a learning rate of 0.04, maximum depth of 3, row and feature subsampling, and L2 regularization.

Saved test results:

| Metric | XGBoost baseline |
|---|---:|
| Accuracy | 0.696 |
| Precision | 0.469 |
| Recall | 0.161 |
| F1-score | 0.240 |
| ROC-AUC | 0.631 |

## 9. Model Comparison with Cross-Validation

A single validation split can depend on how the data is divided. The notebook uses **5-fold StratifiedKFold** to compare Gradient Boosting and XGBoost on the training set.

The comparison includes:

- Mean ROC-AUC
- Standard deviation of ROC-AUC
- Mean F1-score
- Mean recall

Saved cross-validation results:

| Model | Mean ROC-AUC | ROC-AUC Std. | Mean F1 | Mean Recall |
|---|---:|---:|---:|---:|
| XGBoost | 0.679 | 0.021 | 0.302 | 0.212 |
| Gradient Boosting | 0.674 | 0.010 | 0.244 | 0.158 |

The mean score summarizes performance across folds. The standard deviation provides information about variation between folds. Consider both average performance and stability.

## 10. Hyperparameter Tuning with GridSearchCV

The notebook uses `GridSearchCV` with a compact parameter grid for XGBoost.

The search evaluates combinations of:

- `n_estimators`: 150 and 250
- `max_depth`: 2 and 3
- `learning_rate`: 0.03 and 0.07
- `min_child_weight`: 2 and 4

The search uses 5-fold stratified cross-validation and `roc_auc` as its scoring metric. The best estimator is refitted on the full training set.

Saved search result:

| Item | Result |
|---|---|
| Best `learning_rate` | 0.07 |
| Best `max_depth` | 2 |
| Best `min_child_weight` | 4 |
| Best `n_estimators` | 150 |
| Best mean CV ROC-AUC | 0.679 |

## 11. Final Evaluation of the Tuned XGBoost Model

After model selection, the notebook evaluates the selected estimator on the held-out test set.

Saved classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Normal risk (0) | 0.72 | 0.94 | 0.81 | 337 |
| High risk (1) | 0.47 | 0.13 | 0.21 | 143 |
| Overall accuracy | — | — | 0.70 | 480 |
| Macro average | 0.60 | 0.54 | 0.51 | 480 |
| Weighted average | 0.65 | 0.70 | 0.63 | 480 |

**Test ROC-AUC: 0.637**

### Business interpretation

The tuned model performs much better on normal-risk orders than on high-risk orders at the default threshold. Its high-risk recall is only 0.13, which means that most high-risk orders are not identified by the default classification decision.

This is an important reminder: a model can have an accuracy near 70% while still being weak at identifying the positive class that matters to the business.

## 12. Feature Importance and Interpretation

The notebook explores two approaches.

### XGBoost feature importance

Tree-based models provide feature-importance measures that describe how the fitted model uses input features.

These values can help identify features that contribute strongly to the model, but they do not prove that a feature causes return risk. Correlated features can share or distort importance, and impurity-based importance can favor features with more possible split points.

### Permutation importance

Permutation importance measures how much a selected evaluation score decreases when one feature is randomly shuffled.

The notebook uses:

- `permutation_importance()`
- `scoring="roc_auc"`
- `n_repeats=10`

A large decrease suggests that the model relies on that feature for the selected score. This is still a measure of model reliance, not a causal effect.

## 13. Threshold Selection and Operational Trade-offs

The default binary classification threshold is commonly 0.5. A different threshold may be more appropriate for a business workflow.

- Lower threshold: usually flags more orders, which can increase recall and also increase false positives.
- Higher threshold: usually flags fewer orders, which may improve precision in some cases but can miss more high-risk orders.

The notebook compares thresholds of 0.30, 0.40, 0.50, 0.60, and 0.70.

Saved threshold results:

| Threshold | Precision | Recall | F1-score | Flagged orders |
|---:|---:|---:|---:|---:|
| 0.30 | 0.392 | 0.573 | 0.466 | 209 |
| 0.40 | 0.495 | 0.322 | 0.390 | 93 |
| 0.50 | 0.475 | 0.133 | 0.208 | 40 |
| 0.60 | 0.538 | 0.049 | 0.090 | 13 |
| 0.70 | 0.000 | 0.000 | 0.000 | 1 |

**Interpretation:** In this saved run, reducing the threshold from 0.50 to 0.30 increases recall from 0.133 to 0.573 and increases the number of flagged orders from 40 to 209. The team would identify more high-risk orders, but it would also need to review many more orders.

**Validation note:** The notebook's threshold experiment calculates these values on `X_test` and `y_test`. For a strict final evaluation workflow, threshold selection should instead use validation data (or out-of-fold training predictions), and the test set should be used only once after the threshold is fixed. Treat the saved threshold table as a teaching demonstration, not as a final unbiased test comparison.

## 14. Baseline Comparison

A complex model should demonstrate value beyond a simple prediction strategy. The notebook compares the tuned XGBoost model with a `DummyClassifier(strategy="most_frequent")`.

Saved results:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy baseline | 0.702 | 0.000 | 0.000 | 0.000 | 0.500 |
| Tuned XGBoost | 0.698 | 0.475 | 0.133 | 0.208 | 0.637 |

The Dummy baseline has slightly higher accuracy because it predicts the majority class. However, it does not identify any high-risk orders. XGBoost has lower accuracy but provides useful positive-class detection and a ROC-AUC above 0.5.

This comparison shows why model selection should not rely on accuracy alone.

## 15. Applied Mini Project — Return-Risk Model Selection

### Business brief

The Operations Director wants a model that helps the team prioritize orders for manual review before fulfillment.

### Tasks

**Part 1 — Data understanding**
1. Inspect the dataset and target distribution.
2. Explain why each feature could be useful.
3. Identify information that could cause target leakage if collected after the return event.

**Part 2 — Modeling**
1. Train a Gradient Boosting baseline.
2. Train an XGBoost baseline.
3. Use stratified cross-validation on the training set.
4. Tune at least two XGBoost parameters using a compact search.

**Part 3 — Evaluation**
1. Evaluate the selected model on the test set.
2. Report precision, recall, F1-score, and ROC-AUC.
3. Show a confusion matrix and at least one diagnostic curve.
4. Compare the selected model with a simple baseline.

**Part 4 — Interpretation**
1. Show feature importance.
2. Explain why importance does not imply causation.
3. Discuss the impact of changing the classification threshold.

**Part 5 — Recommendation**

Write a short executive recommendation that states:

- Which model you recommend and why.
- Which metric matters most for the proposed workflow.
- What operational risks remain.
- What additional data or monitoring would be needed before deployment.

### Deliverable

Submit the completed notebook with code, outputs, charts, and a one-page business recommendation.

## 16. Knowledge Check

1. How does boosting differ from bagging?
2. Why are trees in Gradient Boosting added sequentially?
3. What is the effect of reducing `learning_rate` while keeping `n_estimators` fixed?
4. Why can increasing `max_depth` increase overfitting risk?
5. Name two regularization-related XGBoost parameters.
6. Why should hyperparameter tuning use cross-validation on the training set?
7. Why should the test set remain untouched during model selection?
8. Why is feature importance not evidence of causation?
9. What operational trade-off can occur when the classification threshold is lowered?
10. Why should a boosting model be compared with a simple baseline?

## 17. How to Use This Notebook

1. Open the notebook using the **Open in Google Colab** badge at the top.
2. Run the notebook from top to bottom.
3. Review the generated dataset and target distribution.
4. Study bagging and boosting before training the first model.
5. Follow the Gradient Boosting experiments.
6. Train and tune XGBoost.
7. Review cross-validation and final test results.
8. Study feature importance and threshold trade-offs.
9. Complete the mini project and knowledge check.

If XGBoost is not installed in a fresh environment, uncomment and run the installation command provided in the notebook.

## 18. Readiness Checklist

Before moving to the next topic, make sure you can:

- [ ] Explain the difference between bagging and boosting.
- [ ] Describe how Gradient Boosting adds trees sequentially.
- [ ] Explain `learning_rate` and `n_estimators`.
- [ ] Explain how tree depth affects model complexity.
- [ ] Describe the main XGBoost parameters.
- [ ] Use stratified cross-validation for model comparison.
- [ ] Tune XGBoost with `GridSearchCV`.
- [ ] Evaluate a model using precision, recall, F1-score, and ROC-AUC.
- [ ] Explain why feature importance does not prove causation.
- [ ] Explain how the classification threshold affects review workload.
- [ ] Compare a trained model with a Dummy baseline.
- [ ] Present a recommendation based on business requirements.

## 19. Key Takeaway

**Gradient Boosting builds trees sequentially to improve the current ensemble, while XGBoost adds regularization and computational optimizations to the boosted-tree approach.** Reliable model development requires controlled complexity, training-only model selection, careful evaluation, and a threshold that reflects the real business cost of classification errors.

## 20. Further Reading

- [scikit-learn — Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- [scikit-learn — GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [scikit-learn — GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [scikit-learn — Permutation Importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
