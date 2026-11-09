<div align="center">

# Machine Learning Fundamentals
### Unit 03 — Underfitting, Overfitting & Model Complexity

**Stage 2 · Machine Learning | Beginner Level**

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TF1qnlv3HYoTB3lNv3TGeRrx3mpK42oS?usp=sharing)

*Understand how model complexity affects predictive performance and learn how to select a model that generalizes well.*

</div>

---

## Overview

How complex should a Machine Learning model be?

A model that is too simple may fail to capture important patterns in the data. A model that is too complex may learn details or noise that do not generalize to new observations.

In this unit, you will explore **underfitting**, **overfitting**, and the relationship between model complexity and predictive performance. Using a Decision Tree Regressor, you will compare different tree sizes and use validation Mean Absolute Error (MAE) to understand the trade-off.

## Business Scenario

You are supporting a property analytics team that wants to estimate Melbourne house prices.

In the previous units, you learned how to prepare the housing data, train a Decision Tree model, and evaluate its predictions using validation data. Now the team needs to understand how the size of the tree affects its ability to make accurate predictions for houses it has not seen before.

Your task is to compare models with different numbers of leaf nodes, identify the strongest candidate among the tested options, and explain the limitations of your selection.

## Dataset

This notebook continues using the provided `melb_data.csv` dataset and the same five numeric input features used in the previous lessons.

| Item | Description |
|---|---|
| Dataset | Melbourne housing data |
| Input features | `Rooms`, `Bathroom`, `Landsize`, `Lattitude`, `Longtitude` |
| Target variable | `Price` |
| Model | Decision Tree Regressor |
| Complexity control | `max_leaf_nodes` |
| Evaluation metric | Mean Absolute Error (MAE) |

The dataset contains historical Melbourne housing records. Results are for learning and should not be treated as current property valuations.

## Learning Outcomes

By the end of this notebook, you should be able to:

- Explain underfitting and overfitting in plain language.
- Describe how tree complexity changes a model's behavior.
- Explain the role of leaf nodes in a Decision Tree Regressor.
- Use `max_leaf_nodes` to control Decision Tree complexity.
- Compare candidate models using validation MAE.
- Identify the best candidate from a defined set of tested options.
- Interpret training and validation errors together.
- Explain why the lowest training error does not necessarily indicate the best model.
- Describe why a validation score is not a guarantee of future performance.
- Explain the role of a separate test set in a responsible model-selection workflow.

## Notebook Roadmap

### 1. Model Complexity and Leaf Nodes
- What a leaf node is
- How the number of leaves affects a Decision Tree
- The relationship between tree size and model complexity

### 2. Underfitting vs. Overfitting
- Understanding underfitting
- Understanding overfitting
- Recognizing common training and validation error patterns
- Understanding the model-complexity trade-off

### 3. Prepare the Dataset
- Load the Melbourne housing dataset
- Select the five input features
- Define the target variable
- Handle missing values in the selected columns
- Create reproducible training and validation sets

### 4. Compare Different Tree Sizes
- Build a reusable evaluation function
- Train models with different `max_leaf_nodes` values
- Calculate validation MAE for each candidate
- Compare candidate model performance

### 5. Visualize and Interpret Validation Results
- Visualize validation MAE across tree sizes
- Identify the lowest validation MAE among tested candidates
- Compare training MAE and validation MAE
- Understand how the gap between the two scores can indicate overfitting

### 6. A Responsible Model-Selection Workflow
- Understand the role of validation data in model selection
- Why the validation score is influenced by repeated model choices
- The purpose of keeping a separate test set untouched
- How cross-validation can support model selection

### 7. Practical Activity and Student Challenge
- Extend the list of candidate leaf counts
- Rerun the model comparison
- Interpret the best candidate
- Write a short recommendation with a result, limitation, and next step

## Key Concepts

| Concept | Meaning |
|---|---|
| Underfitting | The model is too simple to capture important patterns. |
| Good fit | The model captures useful patterns and generalizes reasonably well. |
| Overfitting | The model learns details or noise that do not generalize well to new data. |
| Leaf node | An endpoint of a Decision Tree that contains the prediction for observations reaching it. |
| Model complexity | The degree of flexibility a model has to fit patterns in the training data. |
| `max_leaf_nodes` | A Decision Tree setting that limits the maximum number of leaf nodes. |
| Training MAE | The mean absolute prediction error calculated on training data. |
| Validation MAE | The mean absolute prediction error calculated on validation data. |
| Test set | A separate dataset kept untouched until model choices are complete. |

## Tools and Libraries

- **Python** — programming language
- **Pandas** — data loading and preparation
- **Matplotlib** — visualization
- **scikit-learn** — data splitting, Decision Tree training, prediction, and evaluation
- **Google Colab / Jupyter Notebook** — interactive learning environment

## Practical Activity

Complete the official Kaggle exercise:

[Kaggle — Underfitting and Overfitting](https://www.kaggle.com/code/yazeedalasmar/exercise-underfitting-and-overfitting)

### Student Challenge

1. Add two more values to `candidate_max_leaf_nodes`.
2. Rerun the comparison.
3. Identify the candidate with the lowest validation MAE.
4. Explain why the lowest training MAE is not necessarily the best choice.
5. Write a short recommendation that includes one result, one limitation, and one next step.

### Practice Checklist

- [ ] Explain the difference between underfitting and overfitting.
- [ ] Describe what a leaf node represents.
- [ ] Explain how `max_leaf_nodes` controls tree complexity.
- [ ] Compare validation MAE across the tested candidates.
- [ ] Compare training MAE with validation MAE.
- [ ] Identify the best candidate within the tested list.
- [ ] Explain why the selected candidate is not guaranteed to be optimal.
- [ ] Describe why a separate test set should remain untouched during model selection.

## How to Use This Notebook

1. Click the **Open in Google Colab** badge at the top of this page.
2. Upload `melb_data.csv` to your Colab session when required.
3. Read each Markdown explanation before running its related code cell.
4. Run the code cells in order.
5. Inspect the tables, charts, and evaluation results.
6. Answer the understanding-check questions.
7. Complete the student challenge and write your recommendation.

## Important Learning Notes

- Increasing the number of leaves usually gives a Decision Tree more flexibility, but more complexity does not guarantee better validation performance.
- Training error often decreases as a tree becomes more complex; validation error may stop improving or increase.
- A widening gap between training and validation errors can be a warning sign of overfitting.
- Compare models using the same training/validation split and evaluation metric.
- The best candidate is only the best among the values tested on this particular validation split.
- Removing rows with missing values is a simple teaching choice, not a universal data-preparation strategy.
- The validation set influences model selection. Keep a separate test set untouched until model choices are complete.
- Cross-validation can provide a more robust basis for model selection.

---

<div align="center">

**Machine Learning Fundamentals · Unit 03**

*Find the right balance between learning patterns and generalizing to unseen data.*

</div>
