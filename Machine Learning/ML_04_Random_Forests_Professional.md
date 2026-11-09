<div align="center">

# Machine Learning Fundamentals
### Unit 04 — Random Forests

**Stage 2 · Machine Learning | Beginner Level | B2 English**

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18cGfofrVQYPrr2j4b-nxjIFuP39LKPQT?usp=sharing)

*Learn how combining multiple Decision Trees can improve the stability and generalization of Machine Learning predictions.*

</div>

---

## Overview

A single Decision Tree can be easy to understand, but it may be unstable and can overfit the training data. A natural next step is to combine multiple trees instead of relying on only one.

In this unit, you will learn the main idea behind **Random Forests**, train a `RandomForestRegressor`, and compare its performance with a Decision Tree and a simple baseline. You will also investigate how the number of trees affects validation performance and inspect feature importance.

## Business Scenario

You are supporting a property analytics team that wants to estimate Melbourne house prices.

In the previous units, you trained and evaluated Decision Tree models and explored how model complexity affects generalization. Now the team wants to know whether combining many trees can provide more stable predictions.

Your task is to train a Random Forest, compare it fairly with other candidates, and communicate what the experiment does—and does not—show.

## Dataset

This notebook continues using the supplied `melb_data.csv` dataset and the same target and input features used in the previous units.

| Item | Description |
|---|---|
| Dataset | Melbourne housing data |
| Target (`y`) | `Price` |
| Input features (`X`) | `Rooms`, `Bathroom`, `Landsize`, `Lattitude`, `Longtitude` |
| Main model | `RandomForestRegressor` |
| Comparison models | Decision Tree Regressor and median baseline |
| Evaluation metric | Mean Absolute Error (MAE) |

The dataset contains historical Melbourne housing records. The results are educational examples, not current market valuations.

## Learning Outcomes

By the end of this notebook, you should be able to:

- Explain the main idea behind a Random Forest.
- Describe how Random Forests differ from a single Decision Tree.
- Explain bootstrap sampling and random feature selection at a high level.
- Train a `RandomForestRegressor` using scikit-learn.
- Compare models fairly using the same validation set and MAE.
- Explain what `n_estimators` controls.
- Inspect feature importance and explain its limitations.
- Communicate a model comparison without overstating what the results prove.

## Notebook Roadmap

### 1. Why Random Forests?
- The limitations of a single Decision Tree
- The main idea behind ensemble models
- How a Random Forest combines predictions from many trees
- Bootstrap sampling
- Random feature selection
- Decision Tree vs. Random Forest vs. median baseline

### 2. Prepare the Data
- Load the Melbourne housing dataset
- Select the five numeric input features
- Define the target variable
- Handle missing values in the selected columns
- Create one fixed training/validation split

### 3. Train a Random Forest
- Introduce `RandomForestRegressor`
- Understand the `n_estimators` parameter
- Fit the model using training data
- Generate predictions for validation data
- Interpret Mean Absolute Error (MAE)

### 4. Compare the Random Forest with Other Models
- Compare against a Decision Tree with `max_leaf_nodes=500`
- Compare against a median baseline
- Use the same training/validation split for every candidate
- Visualize and interpret validation MAE
- Discuss what the results support

### 5. Experiment with the Number of Trees
- Understand what `n_estimators` controls
- Compare forests with 50, 100, and 200 trees
- Review validation MAE
- Consider the trade-off between model stability and computation

### 6. Inspect Feature Importance
- Access `feature_importances_`
- Review the relative importance values
- Understand why feature importance is not a causal explanation
- Recognize limitations of impurity-based importance

### 7. Practical Activity and Student Challenge
- Compare another Decision Tree configuration with the Random Forest
- Test an additional number of trees, such as 300
- Identify the strongest candidate in your experiment
- Communicate a result, limitation, and next step

## Key Concepts

| Concept | Meaning |
|---|---|
| Ensemble model | A model that combines predictions from multiple models. |
| Random Forest | An ensemble of Decision Trees whose predictions are combined. |
| Bootstrap sampling | Drawing training samples with replacement for individual trees. |
| Random feature selection | Considering a subset of features when determining tree splits. |
| `n_estimators` | The number of Decision Trees in the forest. |
| Validation set | Held-out data used to compare candidate models. |
| MAE | Average absolute difference between actual and predicted values. |
| Feature importance | A model-derived measure of how features contribute to tree splits; not proof of causation. |
| Baseline | A simple reference model used to judge whether a more complex model adds value. |

## Tools and Libraries

- **Python** — programming language
- **Pandas** — data loading and preparation
- **Matplotlib** — visualization
- **scikit-learn** — data splitting, model training, prediction, and evaluation
- **Google Colab / Jupyter Notebook** — interactive learning environment

## Practical Activity

Complete the corresponding Kaggle exercise:

[Kaggle — Random Forests](https://www.kaggle.com/code/yazeedalasmar/exercise-random-forests)

### Student Challenge

1. Change `max_leaf_nodes` for the Decision Tree and compare it with the Random Forest.
2. Add another `n_estimators` value, such as `300`, and record the validation MAE.
3. Identify the strongest candidate in your experiment.
4. Explain one benefit and one limitation of Random Forests.
5. Write a short recommendation that includes one result, one limitation, and one next step.

### Practice Checklist

- [ ] Explain what an ensemble model is.
- [ ] Describe how a Random Forest combines multiple trees.
- [ ] Explain bootstrap sampling and random feature selection.
- [ ] Train a `RandomForestRegressor`.
- [ ] Compare the Random Forest, Decision Tree, and median baseline.
- [ ] Use the same validation split for all candidates.
- [ ] Compare forests with 50, 100, and 200 trees.
- [ ] Inspect feature importance and explain its limitations.
- [ ] Complete the student challenge.

## How to Use This Notebook

1. Click the **Open in Google Colab** badge at the top of this page.
2. Upload `melb_data.csv` to your Colab session before running the data-loading cell.
3. Read each Markdown explanation before running its related code cell.
4. Run the code cells in order.
5. Inspect the comparison table, charts, and evaluation results.
6. Answer the understanding-check and reflection questions.
7. Complete the student challenge and write your recommendation.

## Important Learning Notes

- A Random Forest often produces more stable predictions than one unconstrained Decision Tree, but it is not guaranteed to achieve a lower validation error in every experiment.
- All candidate models should use the same training/validation split for a fair comparison.
- More trees can make predictions more stable, but they also increase training time and memory use.
- Increasing `n_estimators` does not guarantee a large improvement in validation MAE.
- MAE is expressed in the same unit as the target. In this dataset, the target represents house prices in Australian dollars.
- Feature importance is not a causal explanation and does not prove business impact.
- Dropping rows with missing values is a simple teaching choice, not a universal best practice.
- Keep a separate test set untouched until model selection is complete. Validation results are useful for comparing candidates, but they are not a guarantee of future performance.
- Results may differ from online reference lessons because dataset versions, selected features, data splits, and model settings can differ.

---

<div align="center">

**Machine Learning Fundamentals · Unit 04**

*Move from a single tree to an ensemble—and evaluate the evidence carefully.*

</div>
