[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1N5Xb_Z2JHrsHfjTC5af_4OoYyeDTZzoN?usp=sharing)

# Unit 08 — Cross-Validation

**Stage 2: Machine Learning**  
**Level:** Beginner  
**Language:** English (B2)  
**Business scenario:** Melbourne house-price estimation  
**Environment:** Google Colab / Jupyter Notebook

---

## Overview

A model's validation score can change depending on which rows are selected for a single validation set. A model may perform well on one split and produce a different score on another.

In this unit, we use **five-fold cross-validation** to evaluate a Random Forest pipeline for predicting house prices. Instead of relying on one validation split, we train and evaluate the workflow across five different folds, then summarize the results.

## Business Question

How can we evaluate a machine learning model across several data splits to get a more stable estimate of its performance?

## Learning Outcomes

By the end of this notebook, learners will be able to:

- Explain why a single validation score may depend on the chosen split.
- Describe the steps in k-fold cross-validation.
- Explain the role of training folds and the held-out fold.
- Use `cross_val_score()` with a scikit-learn pipeline.
- Explain why `neg_mean_absolute_error` returns negative values.
- Interpret mean MAE and standard deviation across folds.
- Explain why preprocessing must be fitted inside each training fold.

## Notebook Roadmap

| Section | Topic | What You Will Learn |
|---|---|---|
| 1 | Introduction | Understand why model evaluation is an iterative process and why one validation score may vary. |
| 2 | What Is Cross-Validation? | Learn the k-fold process and how each fold is used for validation once. |
| 2.1 | Read the Diagram | Follow the five experiments in five-fold cross-validation. |
| 2.2 | Why Not Use Only One Validation Set? | Compare the speed and limitations of a holdout split and cross-validation. |
| 3 | Prepare the Data | Load the Melbourne Housing dataset and select the target and numerical features. |
| 3.1 | Inspect Missing Values | Check missing values in the selected features. |
| 4 | Build a Pipeline | Combine mean imputation and a Random Forest model. |
| 5 | Run Cross-Validation | Use `KFold` and `cross_val_score()` to calculate five validation scores. |
| 5.1 | Why Is the Score Negative? | Understand scikit-learn's negative scoring convention. |
| 6 | Interpret the Results | Calculate and visualize mean MAE, standard deviation, minimum, and maximum. |
| 6.1–6.2 | Understand the Scores | Interpret fold variation and understand what cross-validation can and cannot tell us. |
| 7 | Conclusion and Practice | Review the concepts and complete the practical challenge. |

## Key Concepts

### 1. K-Fold Cross-Validation

In k-fold cross-validation, the data is divided into *k* parts called folds. Each experiment uses one fold for validation and the remaining folds for training.

For **five-fold cross-validation**:

- The data is divided into five folds.
- Each experiment uses four folds for training and one fold for validation.
- The process runs five times.
- Every row is used in a validation fold once.

The average of the fold scores provides a summary of model performance across the different splits.

### 2. Holdout Validation vs. Cross-Validation

| Approach | Advantage | Limitation |
|---|---|---|
| One validation split | Faster and simple | The score may depend on the selected split. |
| Cross-validation | Evaluates the model across multiple splits | Requires more model training. |

Cross-validation can be useful for small or medium-sized datasets when model training is reasonably fast. For very large datasets or expensive models, a single holdout set may be more practical.

### 3. The Pipeline Used in This Unit

The notebook creates a scikit-learn `Pipeline` with two steps:

1. **Preprocessor:** `SimpleImputer()` replaces missing numerical values using the mean of each feature.
2. **Model:** `RandomForestRegressor` learns patterns from the prepared data and predicts house prices.

The Random Forest configuration in the notebook uses `n_estimators=50`, `random_state=0`, and `n_jobs=-1`.

### 4. Five-Fold Cross-Validation with `KFold`

The notebook uses an explicit `KFold` splitter:

- `n_splits=5`
- `shuffle=True`
- `random_state=0`

This creates five folds, shuffles the rows before splitting, and makes the split reproducible.

The `cross_val_score()` function runs the pipeline on each fold and returns one score per fold.

### 5. Why Does `neg_mean_absolute_error` Return Negative Values?

Scikit-learn follows a scoring convention where higher scores are better. Since lower Mean Absolute Error is better, the scoring name is `neg_mean_absolute_error`.

The raw scores returned by `cross_val_score()` are negative. The notebook multiplies them by `-1` to convert them into ordinary positive MAE values.

- Lower positive MAE means smaller average absolute prediction errors.
- Compare models using the same folds and the same metric.
- Do not compare MAE values from different target units as if they were equivalent.

### 6. Interpreting the Results

The notebook calculates and displays:

- **Mean MAE:** the average validation error across the five folds.
- **Standard deviation:** how much the fold scores vary around their average.
- **Lowest fold MAE:** the smallest error among the folds.
- **Highest fold MAE:** the largest error among the folds.
- **Range:** the difference between the highest and lowest fold MAE.

A higher fold MAE means larger average absolute errors on that fold. The standard deviation helps describe variation between folds; it is **not** a confidence interval.

## Dataset and Features

**Required file:** `melb_data.csv`  
**Dataset:** Historical Melbourne housing records  
**Target variable:** `Price`

The notebook uses five numerical features:

- `Rooms`
- `Distance`
- `Landsize`
- `BuildingArea`
- `YearBuilt`

Some selected features contain missing values. The pipeline handles them through mean imputation during each cross-validation experiment.

> **Before running the notebook:** Upload `melb_data.csv` to your Google Colab session so the data-loading cell can find it.

The dataset contains historical property records and is used for learning and experimentation, not for current property valuations.

## Data Leakage and Cross-Validation

Preprocessing steps that learn values from data must be fitted on the training portion only.

In this notebook, the imputer is inside the pipeline. During each fold:

1. The imputer learns feature means from that fold's training rows.
2. It uses those learned means to transform the held-out validation fold.
3. The Random Forest is trained and evaluated using the correctly prepared data.

This prevents information from the held-out fold from influencing the imputation values. Fitting preprocessing on the entire dataset before cross-validation could cause data leakage.

**Important:** Cross-validation is normally used during model development. Keep a separate test set for a final, unbiased evaluation when the project requires one.

## Tools and Libraries

The notebook uses:

- **Pandas** — load and inspect the dataset.
- **Matplotlib** — visualize MAE values across folds.
- **scikit-learn**
  - `KFold`
  - `cross_val_score`
  - `Pipeline`
  - `SimpleImputer`
  - `RandomForestRegressor`

## Reference Results

The notebook includes historical reference results from the original Kaggle lesson:

| Fold | Reference MAE |
|---|---:|
| Fold 1 | AUD 301,629 |
| Fold 2 | AUD 303,164 |
| Fold 3 | AUD 287,298 |
| Fold 4 | AUD 236,062 |
| Fold 5 | AUD 260,383 |
| **Average** | **Approximately AUD 277,707** |

These are reference figures, not required outputs. Your results may differ because of dataset versions, selected features, model settings, split strategy, and library versions. Use the values generated by your own notebook run when discussing your experiment.

## How to Use This Notebook

1. Open the notebook using the **Open in Google Colab** button at the top of this page.
2. Upload `melb_data.csv` to the Colab session.
3. Run the notebook cells in order.
4. Review the selected features and missing-value counts.
5. Follow how the pipeline is fitted and evaluated across five folds.
6. Review the MAE for each fold, then interpret the mean and standard deviation.
7. Complete the practical challenge and readiness checklist.

## Practical Challenge

After completing the notebook:

1. Run five-fold cross-validation and record the average MAE.
2. Change `n_estimators` from `50` to `100` and compare the average MAE.
3. Compare the standard deviation for both experiments.
4. Explain why we use a pipeline for preprocessing during cross-validation.
5. Explain why a lower MAE indicates better predictive performance.

## Readiness Checklist

- [ ] I can explain what a fold is.
- [ ] I can describe the five experiments in five-fold cross-validation.
- [ ] I can use `cross_val_score()`.
- [ ] I can explain the negative MAE scoring convention.
- [ ] I can calculate and interpret average MAE.
- [ ] I can explain why preprocessing belongs inside the pipeline.

## Key Takeaway

**Cross-validation evaluates a model across multiple data splits, giving a more informed picture of its performance than a single validation split alone.**

## References

- [Kaggle — Intermediate Machine Learning: Cross-Validation](https://www.kaggle.com/learn/intermediate-machine-learning/lesson/5)
- [scikit-learn — Cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn — `cross_val_score()` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
- [scikit-learn — Common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
