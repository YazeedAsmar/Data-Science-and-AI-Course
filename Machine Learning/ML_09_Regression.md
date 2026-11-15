[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pNig5CdSPU5A19PxvzyrjL1fpiCPZYPi?usp=sharing)

# Unit 09 — Regression: Predicting a Number

**Stage 2: Machine Learning**  
**Level:** Beginner  
**Language:** English (B2)  
**Business scenario:** AdventureWorks order revenue estimation  
**Environment:** Google Colab / Jupyter Notebook

---

## Overview

In this unit, learners explore **regression**, a supervised machine learning task used to predict numerical values.

Using AdventureWorks order data, the goal is to estimate an order's `net_revenue` from information available when the prediction is made. Learners begin with a simple baseline, build and interpret Linear Regression models, evaluate prediction errors, inspect residuals, and combine preprocessing and modeling in a scikit-learn pipeline.

The unit also highlights an important limitation: a good evaluation score does not automatically mean that a model represents the business process correctly.

## Business Question

Can we estimate an order's net revenue before the order is completed, using the order details available at prediction time?

### Prediction Inputs

- `quantity`
- `unit_price`
- `discount_pct`
- `shipping_fee`
- `customer_segment`
- `sales_channel`
- `product_category`

**Target (`y`):** `net_revenue`, a numerical value in dollars.

**Excluded field:** `order_id` is an identifier, not a feature that explains revenue. The target `net_revenue` must never be included among the input features.

Only information known at prediction time should be used. Information created after an order is completed must not be included in the model inputs.

## Learning Outcomes

By the end of this unit, learners will be able to:

- Distinguish regression from classification.
- Identify the target variable and prediction-time features.
- Explain why identifiers should not be treated as predictive features.
- Establish a baseline using the training target mean.
- Explain the idea of fitting a Linear Regression line using least squares.
- Interpret coefficients and the intercept in a regression model.
- Calculate and interpret MAE, RMSE, and R².
- Compare a baseline model, a quantity-only model, and a model using numerical features.
- Interpret residuals and actual-versus-predicted plots.
- Build a preprocessing and modeling workflow using `ColumnTransformer` and `Pipeline`.
- Recognize when a linear model may be too simple for the underlying business relationship.

## Notebook Roadmap

| Section | Topic | What You Will Learn |
|---|---|---|
| 1 | Define the Regression Problem | Identify the business question, target, input features, and prediction moment. |
| 2 | Avoid Invalid Inputs | Understand why `order_id` is excluded and why the target cannot be used as an input. |
| 3 | Establish a Baseline | Use the training mean as a simple prediction and evaluate it with MAE. |
| 4 | Linear Regression | Understand the fitted line, its coefficients, and the least-squares idea. |
| 5 | Interpret Coefficients | Read the estimated relationship between input features and predicted revenue. |
| 6 | Regression Metrics | Understand MAE, RMSE, and R² and when each metric is useful. |
| 7 | Compare Model Performance | Compare the baseline, quantity-only model, and numerical-features model. |
| 8 | Inspect Prediction Errors | Read residuals and actual-versus-predicted plots to identify patterns. |
| 9 | Build a Preprocessing Pipeline | Combine numerical and categorical preprocessing with Linear Regression. |
| 10 | Understand Model Limitations | Explore why a straight-line model may not represent a multiplicative revenue formula. |
| 11 | Regression Checklist | Review the main steps for building and evaluating a regression model. |

## Key Concepts

### 1. Regression vs. Classification

Regression predicts a numerical value, such as an order's net revenue. Classification predicts a category or class, such as `Low`, `Medium`, or `High`.

In this unit, the model predicts a dollar amount rather than choosing from a fixed set of classes.

### 2. Start with a Baseline

A baseline provides a simple reference point. The unit uses a mean baseline that always predicts the mean `net_revenue` calculated from the training data.

Reference result shown in the slides:

- Training mean prediction: **$993**
- Test MAE: **$668**

A more complex model should be compared with this baseline. If it cannot improve on a simple reference model, its added complexity may not be useful.

### 3. Linear Regression and Least Squares

For a single input feature, the Linear Regression equation is:

\[
\hat{y} = b_0 + b_1 x
\]

Where \(\hat{y}\) is the predicted value, \(b_0\) is the intercept, \(b_1\) is the feature coefficient, and \(x\) is the input value.

The least-squares method finds the line that minimizes the sum of squared residuals.

The quantity-only example in the slides shows:

\[
\hat{y} = 46.30 + 210.18 \times quantity
\]

The fitted line is an estimate based on observed data. It does not prove that increasing quantity causes revenue to increase by that amount in every situation.

### 4. Reading Coefficients

A coefficient describes the model's estimated change in prediction for a one-unit increase in that feature, while the other included features are held constant.

The slides show these example coefficients for a multiple-feature model:

| Feature | Coefficient | Interpretation |
|---|---:|---|
| `quantity` | +219.30 | About $219.30 higher prediction per additional unit. |
| `unit_price` | +4.22 | About $4.22 higher prediction per additional $1 of unit price. |
| `discount_pct` | -957.08 | About $95.70 lower prediction for a 0.10 increase in discount. |
| `shipping_fee` | +0.21 | About $0.21 higher prediction per additional $1 of shipping fee. |

The intercept is **-917.84**. It represents the model's prediction when all included inputs are zero, which is not a realistic order in this scenario.

**Important:** Coefficients describe associations learned from the data. They do not establish cause and effect.

### 5. Regression Evaluation Metrics

| Metric | What It Measures | Better Result |
|---|---|---|
| **MAE** | Average absolute prediction error, in dollars. | Lower |
| **RMSE** | Square root of the average squared error; larger errors receive more weight. | Lower |
| **R²** | Compares the model with a model that always predicts the mean. | Higher |

R² can be negative when the model performs worse than the mean-prediction reference on the evaluated data.

### 6. Compare Models Against the Baseline

The slide comparison reports these test results:

| Model | Test MAE |
|---|---:|
| Mean baseline | $668.38 |
| Quantity only | $489.59 |
| Numerical features | $217.45 |

The slides also report:

| Metric | Mean Baseline | Quantity Only | Numerical Features |
|---|---:|---:|---:|
| R² | -0.00 | 0.41 | 0.86 |
| RMSE | 794 | 609 | 292 |

In this experiment, using the numerical features gives a substantially lower test MAE than the baseline and quantity-only model. These are reference results from the presentation; actual results can vary with the data and execution environment.

### 7. Residual Analysis

A residual is the difference between the observed value and the predicted value:

\[
Residual = y - \hat{y}
\]

- **Positive residual:** actual revenue is higher than the prediction.
- **Negative residual:** the prediction is higher than actual revenue.

Residual plots help reveal patterns that a single metric cannot show. The presentation highlights a curved pattern in the residuals, suggesting that the straight-line model may be missing structure in the data.

An ideal residual plot should not show a clear systematic pattern. Patterns can indicate that the model form or selected features are not capturing the relationship adequately.

### 8. One Pipeline for Numerical and Categorical Features

The unit presents a workflow that combines preprocessing and Linear Regression:

| Feature Group | Preprocessing Steps |
|---|---|
| Numerical columns (4) | `SimpleImputer(strategy="median")` → `StandardScaler` |
| Categorical columns (3) | `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder` |

`ColumnTransformer` applies the correct preprocessing steps to each feature group. `Pipeline` then connects the preprocessing workflow to the Linear Regression model.

The presentation reports:

- Full pipeline MAE: **$216.23**
- Numerical-features-only MAE: **$217.45**

The same fitted preprocessing rules are reused for new orders. This helps keep training and prediction transformations consistent.

### 9. A Limitation of a Straight-Line Model

The presentation highlights that revenue is based on a multiplicative relationship:

\[
Revenue = quantity \times price \times (1 - discount)
\]

A simple linear model that adds feature contributions may not represent this relationship well in every case. The slides show an example where a Linear Regression prediction is negative, even though negative revenue is not possible in the business context.

An additional experiment is shown in the presentation: create a feature named `gross` using the revenue expression above. The slide reports an MAE change from **217 to 47**.

> **Important:** The presentation explicitly labels this as an extra experiment, not part of the notebook. Treat it as an extension idea rather than an output learners should expect from the notebook.

## Tools and Libraries

The presentation identifies a scikit-learn workflow using:

- **Linear Regression** — predict a numerical target.
- **Baseline model** — provide a simple reference prediction.
- **SimpleImputer** — fill missing values.
- **StandardScaler** — scale numerical features.
- **OneHotEncoder** — encode categorical features.
- **ColumnTransformer** — apply different preprocessing to different feature groups.
- **Pipeline** — connect preprocessing and model training.
- **MAE, RMSE, and R²** — evaluate regression performance.

## How to Use This Notebook

1. Open the notebook using the **Open in Google Colab** button at the top of this page.
2. Run the notebook cells in order.
3. Identify the target variable and the information available at prediction time.
4. Review the baseline before evaluating more complex models.
5. Compare the regression models using the evaluation metrics.
6. Inspect the residual and actual-versus-predicted plots.
7. Review how numerical and categorical preprocessing is combined in a pipeline.
8. Use the presentation's additional `gross` feature experiment as an extension discussion, not as a required notebook result.

## Practical Challenge

After completing the notebook:

1. Compare the mean baseline, quantity-only model, and numerical-features model using MAE.
2. Explain why `order_id` should not be included as a model feature.
3. Interpret one coefficient from the multiple-feature model.
4. Explain what a positive and a negative residual mean.
5. Compare MAE and RMSE. Why does RMSE give more weight to large errors?
6. Explain why the pipeline uses different preprocessing steps for numerical and categorical columns.
7. Discuss why a linear model may produce an unrealistic negative revenue prediction.

## Readiness Checklist

- [ ] I can explain the difference between regression and classification.
- [ ] I can identify the target and valid prediction-time features.
- [ ] I understand why a baseline is important.
- [ ] I can explain the Linear Regression equation and least-squares idea.
- [ ] I can interpret coefficients and the intercept.
- [ ] I can compare MAE, RMSE, and R².
- [ ] I can read residuals and identify patterns in a residual plot.
- [ ] I understand how `ColumnTransformer` and `Pipeline` work together.
- [ ] I can identify limitations of a simple linear model.

## Key Takeaway

**A regression model predicts a number, but building a useful regression solution requires more than fitting a line: define valid inputs, compare against a baseline, evaluate errors, inspect residuals, and keep preprocessing consistent.**

## References

- [scikit-learn — Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [scikit-learn — Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [scikit-learn — ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [scikit-learn — Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
