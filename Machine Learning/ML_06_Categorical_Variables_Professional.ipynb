[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1g63f0yiaGwuL-x7woh6gsXxzFbc2lZzr?usp=sharing)

# Unit 06 — Categorical Variables

### Stage 2: Machine Learning | Data Science & AI Program

**Level:** Beginner  
**Dataset:** Melbourne Housing (`melb_data.csv`)  
**Environment:** Google Colab / Jupyter Notebook  
**Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn

---

## Overview

Real-world datasets contain more than numbers. They also include categories such as property type, suburb, seller, and other labels.

Most machine learning estimators in Scikit-learn require input features to be represented numerically. This unit explains how to prepare categorical data for machine learning while avoiding misleading numeric relationships and data leakage.

Using the Melbourne Housing dataset, we compare three approaches with the same training/validation split and Random Forest model:

1. Drop categorical variables.
2. Apply ordinal encoding.
3. Apply one-hot encoding.

**Business question:** How can we convert categorical values into a format that a machine learning model can use without losing useful information or introducing misleading relationships?

> **Dataset setup:** Upload `melb_data.csv` to your Google Colab session before running the data-loading cell. The notebook uses historical housing data for teaching; its results are not current market valuations.

---

## Learning Outcomes

By the end of this unit, you should be able to:

- Distinguish between ordinal and nominal categorical variables.
- Explain why many machine learning estimators cannot directly use text categories.
- Compare dropping categorical features, ordinal encoding, and one-hot encoding.
- Use Scikit-learn's `OrdinalEncoder` and `OneHotEncoder`.
- Handle categories that appear in validation data but not in training data.
- Compare preprocessing strategies using validation Mean Absolute Error (MAE).
- Explain why preprocessing must be fitted on training data only to avoid data leakage.

---

## Notebook Roadmap

| Section | Topics |
|---|---|
| 1. Introduction | What categorical variables are; ordinal vs. nominal variables |
| 2. Three Approaches | Dropping categories, ordinal encoding, and one-hot encoding |
| 2.1 Quick Comparison | Strengths and suitable use cases for each approach |
| 3. Prepare the Data | Target separation, train/validation split, and feature selection |
| 4. Evaluate the Approaches | Train a Random Forest model and compare validation MAE |
| 5. Review | Compare results, limitations, and review questions |
| 6. Conclusion and Practice | Key takeaways, Kaggle practice, and a practical challenge |

---

## Key Concepts Covered

### 1. Types of Categorical Variables

- **Ordinal variables:** Categories have a meaningful order. Example: `Never < Rarely < Most days < Every day`.
- **Nominal variables:** Categories have no natural ranking. Example: `Red`, `Yellow`, and `Green`.

The distinction matters because assigning numbers to nominal categories can suggest an order that does not actually exist.

### 2. Categorical Encoding Methods

| Method | How it works | Suitable when |
|---|---|---|
| Drop categorical variables | Removes categorical columns from the dataset | Categories provide little useful information |
| Ordinal encoding | Maps each category to an integer | Categories have a meaningful order |
| One-hot encoding | Creates a binary column for each category | Categories have no natural order and cardinality is manageable |

The notebook also introduces **cardinality**: the number of unique categories in a feature. High-cardinality features can create many columns when one-hot encoded.

### 3. Scikit-learn Encoders

The practical work uses:

- `OrdinalEncoder` to convert categories into integer codes.
- `OneHotEncoder` to create binary indicator columns.
- `handle_unknown='ignore'` to avoid errors when validation data contains a category that was not present during training.
- `sparse_output=False` to return a dense array for this introductory example. Sparse output may use less memory for larger datasets.

### 4. Model Evaluation

The notebook defines `score_dataset()` to train a Random Forest model and calculate validation MAE.

A lower MAE indicates smaller average absolute prediction errors on the validation data.

---

## Experiment Results

The notebook reports the following reference validation MAE values:

| Approach | Reference validation MAE |
|---|---:|
| Drop categorical variables | $175,703 |
| Ordinal encoding | $165,936 |
| One-hot encoding | $166,089 |

**Interpretation:**

- Dropping categorical columns produced the highest MAE in the original example.
- Ordinal encoding achieved the lowest MAE in this experiment.
- One-hot encoding produced a very similar result to ordinal encoding.
- The small difference between the two encoding methods does not establish that one method is always better.

> These are reference results from the original lesson. Your results may differ depending on the dataset version and software environment. Run the notebook and use your actual validation results when discussing model performance.

---

## Practical Activities

During this unit, learners will:

1. Inspect the Melbourne Housing dataset and identify categorical columns.
2. Separate the target variable (`Price`) from the predictors.
3. Prepare training and validation subsets.
4. Compare the model after dropping categorical columns.
5. Apply ordinal encoding and evaluate the result.
6. Apply one-hot encoding and handle unknown validation categories.
7. Compare the three approaches using validation MAE.
8. Interpret why removing categorical information may reduce model performance.

### Review Questions

- What makes a variable categorical?
- What is the difference between ordinal and nominal variables?
- Why can ordinal encoding be misleading for categories such as color or region?
- What does `handle_unknown='ignore'` do in `OneHotEncoder`?
- Why should an encoder be fitted on training data and then used to transform validation data?
- Which approach achieved the lowest MAE in the original example?

### Practical Challenge

1. Run all three approaches and record their MAE scores.
2. Identify the approach with the lowest validation MAE.
3. Explain why dropping categorical variables may produce a worse result.
4. Choose an ordinal variable and explain why its categories have a meaningful order.
5. Try a feature with many unique categories and discuss how one-hot encoding changes the number of columns.

---

## Tools & Libraries

- **Python** — Main programming language.
- **Pandas** — Data loading and data preparation.
- **NumPy** — Numerical operations.
- **Matplotlib** — Basic data visualization.
- **Scikit-learn** — Encoders, Random Forest model, and evaluation workflow.
- **Google Colab / Jupyter Notebook** — Interactive learning and experimentation.

---

## How to Use This Notebook

1. Open the notebook using the **Open in Google Colab** badge at the top of this README.
2. Upload `melb_data.csv` to the Colab session when requested by the notebook.
3. Run the cells in order, from top to bottom.
4. Read each explanation before running its code.
5. Compare the MAE values produced by the three approaches.
6. Complete the review questions and practical challenge.

---

## Important Learning Notes

- Do not treat the integer codes created by ordinal encoding as meaningful rankings when the original categories are nominal.
- One-hot encoding is often a good starting point for nominal categories, but high cardinality can create many additional columns.
- There is no single encoding method that is best for every dataset and model.
- Compare preprocessing strategies using validation data rather than assuming that one method is always superior.
- Fit preprocessing steps on the training data only, then apply the fitted transformation to validation data. This helps prevent data leakage.

---

## Additional Practice & References

- [Kaggle — Intermediate Machine Learning: Categorical Variables](https://www.kaggle.com/learn/intermediate-machine-learning/lesson/3)
- [Kaggle — Full Intermediate Machine Learning Course](https://www.kaggle.com/learn/intermediate-machine-learning)
- [Scikit-learn — OrdinalEncoder Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
- [Scikit-learn — OneHotEncoder Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

---

## Key Takeaway

**Categorical data must be represented in a form that a machine learning model can use. Choose the encoding method based on the meaning of the categories, then compare its performance on validation data.**
