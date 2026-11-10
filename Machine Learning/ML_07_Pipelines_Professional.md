[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZzxZ3IS00qCwbzIhlZlRh6sebtjkLQA2?usp=sharing)

# Unit 07 — End-to-End Preprocessing with Pipelines

**Stage 2: Machine Learning**  
**Level:** Beginner  
**Language:** English (B2)  
**Business scenario:** Melbourne house-price estimation  
**Environment:** Google Colab / Jupyter Notebook

---

## Overview

Real-world datasets often contain missing values and a mix of numerical and categorical features. Preparing these columns manually can lead to inconsistent transformations and data leakage.

In this unit, we build a repeatable machine learning workflow using **scikit-learn's `ColumnTransformer` and `Pipeline`**. We apply different preprocessing steps to numerical and categorical columns, then connect the preprocessing workflow to a Random Forest model.

The goal is to train, evaluate, and reuse the complete workflow as one pipeline.

## Business Question

How can we prepare numerical and categorical housing data consistently, train a house-price prediction model, and apply the same preprocessing steps to validation or new data without repeating the work manually?

## Learning Outcomes

By the end of this unit, learners will be able to:

- Explain why preprocessing should be part of the model workflow.
- Separate numerical and categorical features.
- Use `ColumnTransformer` to apply different transformations to different column groups.
- Handle missing numerical values with median imputation.
- Handle missing categorical values with most-frequent imputation.
- Apply one-hot encoding to categorical features.
- Combine preprocessing and a Random Forest model using `Pipeline`.
- Fit and evaluate the complete workflow using validation MAE.
- Explain how pipelines help reduce inconsistent preprocessing and data leakage.
- Use a fitted pipeline to predict a new example.

## Notebook Roadmap

| Section | Topic | What You Will Do |
|---|---|---|
| 1 | Introduction to Pipelines | Understand the purpose and benefits of a pipeline. |
| 2 | Prepare the Data | Load the Melbourne Housing dataset, separate the target, split the data, and identify feature types. |
| 3 | Define Preprocessing Steps | Use `ColumnTransformer` and separate numerical and categorical preprocessing paths. |
| 4 | Define the Model | Create a `RandomForestRegressor`. |
| 5 | Create and Evaluate the Pipeline | Connect preprocessing and modeling, fit the workflow, and evaluate validation MAE. |
| 5.1 | Understand `fit()` | Follow the operations performed when the pipeline is trained. |
| 5.2 | Inspect Transformed Features | View the feature names and number of columns created by preprocessing. |
| 5.3 | Predict a New Example | Use the fitted pipeline to predict the price of a property example. |
| 5.4 | Data Leakage | Understand why preprocessing must be learned from training data only. |
| 6 | Conclusion and Practice | Review the key ideas and complete the practical challenge. |

## Key Concepts

### 1. Machine Learning Pipeline

A pipeline connects preprocessing and model training into one workflow:

`Raw Features → Preprocessing → Model → Prediction`

It helps keep code organized, reduces preprocessing mistakes, makes the workflow easier to reuse, and supports safer model validation.

### 2. ColumnTransformer

`ColumnTransformer` applies different transformations to selected groups of columns.

| Feature Type | Preprocessing in This Unit |
|---|---|
| Numerical | Fill missing values using the median. |
| Categorical | Fill missing values using the most frequent category, then apply one-hot encoding. |

The notebook selects categorical columns with fewer than 10 unique values. This is a simple rule for this introductory lesson, not a universal feature-selection method.

### 3. Pipeline

The notebook uses `Pipeline` to connect the preprocessing stage to a `RandomForestRegressor`.

When the pipeline is fitted, it learns the required imputation values and category mappings from the training data, transforms the features, and trains the model. During prediction, it reuses those learned transformations automatically.

### 4. Data Leakage Prevention

Preprocessing steps that learn information from data should be fitted on the training set only. For example, calculating imputation values before splitting the dataset can allow information from validation rows to influence preprocessing.

Keeping preprocessing inside the pipeline helps ensure that the validation data is not used to learn imputation values or category mappings.

## Tools and Libraries

The notebook uses:

- **Pandas** — load and prepare the dataset.
- **NumPy** — create baseline predictions.
- **Matplotlib** — visualize the MAE comparison.
- **scikit-learn**
  - `train_test_split`
  - `ColumnTransformer`
  - `Pipeline`
  - `SimpleImputer`
  - `OneHotEncoder`
  - `RandomForestRegressor`
  - `mean_absolute_error`

## Model Evaluation

The complete pipeline is evaluated using **Mean Absolute Error (MAE)** on the validation set.

The notebook also compares the pipeline with a simple baseline that always predicts the median house price from the training data.

- Lower MAE indicates a lower average absolute prediction error on the validation set.
- The notebook references an original Kaggle example MAE of approximately **AUD 160,679** under its specific data and preprocessing setup.
- Your result may differ because of dataset versions, library versions, selected features, and preprocessing choices.

The notebook displays the result from your current run, so use that value when discussing your experiment. Neither validation score guarantees performance on future properties.

## Dataset

**File required:** `melb_data.csv`  
**Dataset:** Historical Melbourne housing records  
**Target variable:** `Price`

The notebook uses selected numerical columns and categorical columns with fewer than 10 unique values.

> **Before running the notebook:** Upload `melb_data.csv` to your Google Colab session so the data-loading cell can find it.

This dataset is used for learning and experimentation. The model is not intended to provide current property valuations.

## How to Use This Notebook

1. Open the notebook using the **Open in Google Colab** button at the top of this page.
2. Upload `melb_data.csv` to the Colab session.
3. Run the notebook cells in order.
4. Review the selected numerical and categorical columns.
5. Follow how the preprocessing steps are combined with the Random Forest model.
6. Record the validation MAE and compare it with the training-median baseline.
7. Complete the practical challenge and practice checklist.

## Practical Challenge

After completing the notebook:

1. Run the complete pipeline and record its validation MAE.
2. Experiment with a different numerical imputation strategy.
3. Try another model, such as `DecisionTreeRegressor`, while keeping the preprocessing steps unchanged.
4. Compare the validation MAE values.
5. Explain why preprocessing should be fitted on training data only.

**Note for learners:** The current notebook already uses median imputation for numerical features. Therefore, when experimenting with imputation, choose and compare another strategy rather than repeating the median setting.

## Readiness Checklist

- [ ] I can explain what a machine learning pipeline does.
- [ ] I can use `ColumnTransformer`.
- [ ] I can define separate numerical and categorical preprocessing steps.
- [ ] I can combine preprocessing and a model with `Pipeline`.
- [ ] I can evaluate a complete pipeline using validation MAE.
- [ ] I can explain how pipelines help prevent data leakage.

## Key Takeaway

**A pipeline connects preprocessing and modeling into one repeatable workflow, making machine learning code cleaner, safer, and easier to reuse.**

## References

- [Kaggle — Intermediate Machine Learning: Pipelines](https://www.kaggle.com/learn/intermediate-machine-learning/lesson/4)
- [scikit-learn — Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html)
- [scikit-learn — SimpleImputer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
- [scikit-learn — OneHotEncoder documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
