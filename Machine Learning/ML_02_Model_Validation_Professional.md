<div align="center">

# Machine Learning Fundamentals
### Unit 02 — Model Validation

**Stage 2 · Machine Learning | Beginner Level**

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17Pg_nfPKHfN636X4ucV9S1YoCet8IC6o?usp=sharing)

*Learn how to evaluate a Machine Learning model using data it has not seen during training.*

</div>

---

## Overview

A Machine Learning model may achieve very good results on the same examples used to train it. However, this does not necessarily mean that it will make accurate predictions for new examples.

In this unit, you will learn why evaluating a model on its training data can be misleading, how to create a separate validation set, and how to use **Mean Absolute Error (MAE)** to estimate predictive performance on unseen data.

## Business Scenario

You are working with a property analytics team that wants to estimate Melbourne house prices.

In Unit 01, you trained a Decision Tree model using historical property records. Now the team needs to understand whether the model can make useful predictions for properties it has not seen before.

Your task is to evaluate the model using a validation set and compare its performance with a simple baseline.

## Dataset

This notebook uses the supplied `melb_data.csv` file, which contains historical Melbourne property records.

| Item | Description |
|---|---|
| Dataset | Melbourne housing data |
| Target variable | `Price` |
| Input features | `Rooms`, `Bathroom`, `Landsize`, `Lattitude`, `Longtitude` |
| Model | Decision Tree Regressor |
| Evaluation metric | Mean Absolute Error (MAE) |

**Note:** This lesson focuses on evaluation methods, not on building the best possible house-price model. The results should be treated as a learning baseline, not as a reliable current property valuation.

## Learning Outcomes

By the end of this lesson, you should be able to:

- Explain why a model must be evaluated on data not used for fitting.
- Calculate and interpret Mean Absolute Error (MAE).
- Distinguish between training performance and validation performance.
- Split features and targets using `train_test_split`.
- Compare training and validation errors using a consistent model setup.
- Explain the purpose of `random_state`.
- Identify why a large gap between training and validation scores may indicate overfitting.
- Compare a trained model with a simple baseline prediction.

## Notebook Roadmap

### 1. What Is Model Validation?
- Why a model must be tested on unseen examples
- The purpose of model validation
- Understanding generalization

### 2. Understand Mean Absolute Error (MAE)
- Actual price versus predicted price
- Calculating absolute prediction errors
- Understanding the MAE formula
- Interpreting MAE in the context of house prices

### 3. Load and Prepare the Melbourne Dataset
- Load the supplied dataset
- Select the input features (`X`)
- Select the target variable (`y`)
- Handle missing values in the selected columns

### 4. The Problem with In-Sample Evaluation
- What an in-sample score means
- Train a Decision Tree using the selected data
- Evaluate the model on the same rows used for training
- Understand why this can produce a misleading result

### 5. Create a Validation Set
- Separate training data from validation data
- Use `train_test_split()`
- Reserve 25% of the data for validation
- Set `random_state=0` for a repeatable split
- Fit the model using training data only
- Generate predictions for the validation set

### 6. Compare and Interpret Results
- Compare training MAE with validation MAE
- Understand the possible signs of overfitting
- Learn what `random_state` does
- Compare the Decision Tree with a simple median baseline
- Interpret what the evaluation scores can and cannot tell us

### 7. Practice and Readiness Check
- Review key model validation concepts
- Complete the practical activity
- Change the random state and compare results
- Explain why evaluation scores may change between different splits

## Key Concepts

| Concept | Meaning |
|---|---|
| Training set | Data used to fit the model |
| Validation set | Held-out data used to estimate model performance and compare model choices |
| In-sample evaluation | Evaluating a model on the same examples used to fit it |
| MAE | Average absolute difference between actual and predicted values |
| Overfitting | A model fits training examples well but performs less effectively on unseen data |
| `random_state` | Controls the random split so the result can be reproduced |
| Baseline | A simple reference prediction used to judge whether the model adds value |

## Tools and Libraries

- **Python** — programming language
- **Pandas** — loading and preparing tabular data
- **Matplotlib** — visualization
- **scikit-learn** — data splitting, model training, prediction, and evaluation
- **Google Colab / Jupyter Notebook** — interactive learning environment

## Practical Activity

Complete the corresponding Kaggle lesson:

[Kaggle — Model Validation](https://www.kaggle.com/code/yazeedalasmar/notebook931b454bdd)

### Practice Checklist

- [ ] Select the features and target.
- [ ] Split the data into training and validation sets.
- [ ] Fit the model using only training data.
- [ ] Generate predictions for the validation features.
- [ ] Calculate MAE using validation targets and predictions.
- [ ] Compare training and validation MAE.
- [ ] Compare the Decision Tree against the training-median baseline.
- [ ] Explain what the scores suggest and what they cannot prove.

### Mini-Challenge

Change `random_state` in `train_test_split()` from `0` to `42`, then rerun the split and evaluation.

Compare the scores and explain:

1. Why might the MAE values change?
2. Why does `random_state` support reproducibility?
3. Why should you not keep changing the split just to obtain a preferred score?

## How to Use This Notebook

1. Click the **Open in Google Colab** badge at the top of this page.
2. Upload `melb_data.csv` to your Colab session when required.
3. Read each Markdown explanation before running the related code cell.
4. Run the code cells in order.
5. Inspect the outputs and compare the training and validation results.
6. Answer the understanding-check questions.
7. Complete the practical activity and mini-challenge.

## Important Learning Notes

- A very low training MAE does not prove that a model will perform well on unseen data.
- The validation set must not be used to fit the model.
- Compare training and validation scores using the same model setup.
- A large difference between training and validation MAE may indicate overfitting.
- `random_state` makes a split repeatable; it does not automatically improve model accuracy.
- The median baseline should be calculated from the training targets only.
- One validation split does not guarantee that the same score will occur on every possible sample.
- MAE is expressed in the same unit as the target. Here, the target is house price in the dataset.

---

<div align="center">

**Machine Learning Fundamentals · Unit 02**

*Evaluate beyond the training data. Measure how well your model generalizes.*

</div>
