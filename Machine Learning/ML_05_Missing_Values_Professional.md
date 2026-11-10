<div align="center">

# Machine Learning Fundamentals
### Unit 05 — Missing Values

**Stage 2 · Machine Learning | Beginner Level | B2 English**

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-jHWynlvLvUecM4PNskP4XLCJt0tvG3T?usp=sharing)

*Learn how to identify missing data, compare preprocessing strategies, and evaluate their effect on Machine Learning performance.*

</div>

---

## Overview

Real-world datasets are not always complete. Some records may have values that were not collected, entered, or recorded. Before training a Machine Learning model, these missing values must be handled in a way that is appropriate for the dataset and the task.

In this unit, you will explore three approaches: dropping columns with missing values, imputing missing values, and combining imputation with missingness indicators. You will compare these strategies using the same training/validation split, a Random Forest model, and Mean Absolute Error (MAE).

## Business Scenario

You are supporting a property analytics team that wants to estimate house prices in Melbourne.

The housing dataset contains incomplete property records. The team needs to decide how to handle these gaps without removing useful information or creating misleading model results.

Your task is to inspect the missing data, test different preprocessing strategies, and recommend an approach based on validation performance.

## Dataset

This notebook uses the provided `melb_data.csv` file and the Melbourne Housing dataset.

| Item | Description |
|---|---|
| Dataset | Melbourne housing data |
| Target (`y`) | `Price` |
| Input features (`X`) | Numeric features selected from the dataset |
| Main model | `RandomForestRegressor` |
| Preprocessing approaches | Drop columns, mean imputation, mean imputation with indicators |
| Evaluation metric | Mean Absolute Error (MAE) |
| Dataset file | `melb_data.csv` |

**Important:** Upload `melb_data.csv` to your Google Colab session before running the data-loading cell. The file should be available in the notebook's current working directory.

## Learning Outcomes

By the end of this notebook, you should be able to:

- Explain why missing values occur and why they are not the same as zero.
- Identify missing values and summarize their frequency.
- Compare dropping columns, imputation, and missingness indicators.
- Use `SimpleImputer` to replace missing numeric values.
- Evaluate preprocessing strategies using validation MAE.
- Explain why preprocessing steps should be fitted on training data only.
- Describe how fitting preprocessing on training data only helps prevent data leakage.

## Notebook Roadmap

### 1. Introduction to Missing Values
- Understand common reasons why data may be missing.
- Recognize how missing values are represented as `NaN` in Pandas.
- Understand why a missing value does not mean zero.
- Learn why missing values must be handled before training many Machine Learning models.

### 2. Three Approaches to Missing Values
- **Drop columns:** remove features that contain missing values.
- **Imputation:** replace missing values with a statistic such as the mean.
- **Imputation + indicators:** fill missing values and add indicator columns to record which values were originally missing.
- Compare the advantages and limitations of each approach.

### 3. Prepare and Inspect the Data
- Work with the Melbourne Housing dataset.
- Prepare the target and numeric input features.
- Split the data into training and validation subsets.
- Audit missing-value counts and percentages.
- Identify columns that contain missing values in the training data.
- Understand why the split must happen before fitting an imputer.

### 4. Define a Model Evaluation Function
- Create the `score_dataset()` function.
- Train a Random Forest model using the prepared data.
- Generate predictions on the validation set.
- Calculate Mean Absolute Error (MAE).
- Understand why the same evaluation process is used for all approaches.

### 5. Compare the Three Approaches
- Remove columns with missing values from both training and validation data.
- Apply mean imputation using `SimpleImputer`.
- Add missingness indicator columns after imputation.
- Compare validation MAE for each approach.
- Review the dynamically generated results table and chart.
- Distinguish current notebook results from historical reference scores.

### 6. Interpret the Results
- Understand why preserving useful features may help model performance.
- Review the missing-value counts reported in the original course example.
- Explain why the best approach depends on the dataset and experiment.
- Consider whether median imputation may be worth testing.

### 7. Conclusion and Practical Challenge
- Summarize when dropping, imputing, or adding indicators may be appropriate.
- Compare the approaches using your own validation results.
- Test `strategy='median'` instead of `strategy='mean'`.
- Explain why the imputer is fitted on training data and only transformed on validation data.
- Write a short recommendation for handling missing values in this dataset.

## Key Concepts

| Concept | Meaning |
|---|---|
| Missing value | A value that is not available in the dataset; it is not automatically equal to zero. |
| `NaN` | A common representation of a missing value in Pandas. |
| Dropping columns | Removing features that contain missing values. |
| Imputation | Replacing missing values with an estimated or selected value. |
| `SimpleImputer` | A scikit-learn tool for replacing missing values using a selected strategy. |
| Mean imputation | Replacing missing values with the mean of the corresponding feature. |
| Missingness indicator | An additional feature that records whether a value was missing. |
| Training set | The data used to learn model parameters and preprocessing statistics. |
| Validation set | Held-out data used to evaluate and compare candidate approaches. |
| Data leakage | The use of information during training or preprocessing that would not legitimately be available at prediction time. |
| MAE | The average absolute difference between actual and predicted values; lower values indicate smaller average errors. |

## Reference Results from the Original Lesson

The notebook includes historical reference scores to help interpret the experiment. These are not guaranteed outputs; use the results generated by your own notebook run.

| Approach | Reference MAE | Interpretation |
|---|---:|---|
| Drop columns with missing values | 183,550 | Highest error among the three reference results |
| Mean imputation | 178,166 | Lowest error in this comparison |
| Mean imputation + indicators | 178,928 | Slightly higher error than mean imputation alone |

In the original example, mean imputation performed best among the three approaches. This result is specific to that experiment and does not establish that mean imputation is always the best strategy.

## Tools and Libraries

- **Python** — programming language
- **Pandas** — data loading, inspection, and preparation
- **NumPy** — numerical operations
- **Matplotlib** — visualization
- **scikit-learn** — imputation, Random Forest training, and evaluation
- **Google Colab / Jupyter Notebook** — interactive learning environment

## Practical Activity

Continue your practice with the corresponding Kaggle lesson:

[Kaggle — Intermediate Machine Learning: Missing Values](https://www.kaggle.com/learn/intermediate-machine-learning/lesson/2)

[Open the full Kaggle Intermediate Machine Learning course](https://www.kaggle.com/learn/intermediate-machine-learning)

### Student Challenge

1. Run all three approaches and record their MAE scores.
2. Identify which approach performs best on your validation data.
3. Try `strategy='median'` instead of `strategy='mean'`.
4. Explain why the imputer is fitted on training data and only transformed on validation data.
5. Write a short recommendation for how you would handle missing values in this dataset.

### Practice Checklist

- [ ] Explain why missing values occur and why they are not equal to zero.
- [ ] Identify missing values in a DataFrame.
- [ ] Explain the difference between dropping columns and imputing values.
- [ ] Use `SimpleImputer` to fill missing numeric values.
- [ ] Create missingness indicator columns.
- [ ] Compare preprocessing approaches using validation MAE.
- [ ] Explain how to avoid data leakage during preprocessing.
- [ ] Complete the practical challenge and communicate your recommendation.

## How to Use This Notebook

1. Click the **Open in Google Colab** badge at the top of this page.
2. Upload `melb_data.csv` to your Colab session before running the data-loading cell.
3. Read each Markdown explanation before running its related code cell.
4. Run the code cells in order.
5. Review the missing-value audit and identify the affected features.
6. Run all three preprocessing approaches and inspect the comparison table and chart.
7. Use the MAE values from your current run when discussing the results.
8. Complete the student challenge and write your recommendation.

## Important Learning Notes

- A missing value is not the same as zero. It represents information that is not available.
- Do not drop or replace missing values automatically. First inspect how much data is missing and which features are affected.
- Dropping a column may remove useful information from every record.
- Imputation preserves the feature, but the replacement value may not represent the true missing value.
- Missingness indicators may help when the fact that a value is missing contains useful information, but they do not always improve model performance.
- Split the data before fitting an imputer. Fit the imputer on training data, then use the fitted imputer to transform validation data. This helps prevent data leakage.
- Use the dynamically generated table and chart as the results of your current run. The reference MAE values are historical course examples and may differ from your results.
- In the original example, the training data had 10,864 rows and 12 columns. The reported missing-value counts were 49 for `Car`, 5,156 for `BuildingArea`, and 4,307 for `YearBuilt`.
- A lower validation MAE indicates smaller average absolute errors on that validation set. It does not guarantee the same performance on future data.
- Consider testing other imputation strategies, such as the median, and compare them using a consistent validation process.

## References

- [Kaggle — Intermediate Machine Learning: Missing Values](https://www.kaggle.com/learn/intermediate-machine-learning/lesson/2)
- [scikit-learn — SimpleImputer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
- [scikit-learn — Common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html)

---

<div align="center">

**Machine Learning Fundamentals · Unit 05**

*Inspect the missing data, compare preprocessing strategies, and let validation results guide your decision.*

</div>
