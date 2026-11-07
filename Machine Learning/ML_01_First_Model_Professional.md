"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sv5OWGjoVmFSGEyZtiQ8iN8cNV1y7Gw1?usp=sharing)\n",

# Machine Learning Fundamentals — Unit 01: Your First Machine Learning Model

**Stage 2: Machine Learning | Data Science & AI Program**

---

## Overview

This notebook introduces the basic Machine Learning workflow. You will learn how a model uses examples to identify patterns, explore a dataset, prepare input features and a target variable, train a model, and evaluate its predictions.

## Learning Objectives

By the end of this notebook, you should be able to:

- Explain the basic idea of Machine Learning.
- Distinguish between features and the target variable.
- Explore a tabular dataset using Pandas.
- Prepare data for a Machine Learning task.
- Split data into training and test sets.
- Train a Decision Tree model.
- Generate predictions using a trained model.
- Evaluate regression predictions using Mean Absolute Error (MAE).
- Explain why evaluation on unseen data is important.

## Business Scenario

You are working with a property analytics team that wants to estimate house prices using information about properties. Your task is to explore the available data, select relevant information, build a first predictive model, and evaluate how well it performs.

## Dataset

The notebook uses a Melbourne housing dataset.

- **Prediction target:** `Price`
- **Example input features:** `Rooms`, `Bathroom`, `Landsize`, `Lattitude`, and `Longtitude`

The selected features provide a simple starting point for learning the modeling workflow. They should not be considered the only factors that may influence house prices.

## Notebook Contents

1. **How Models Work**
   - What a Machine Learning model is
   - Features, target, training, and prediction
   - Understanding the basic idea of a Decision Tree

2. **Basic Data Exploration**
   - Inspecting dataset structure
   - Exploring columns and data types
   - Reviewing summary statistics
   - Checking missing values
   - Selecting columns for the first model

3. **Your First Machine Learning Model**
   - Defining `X` (features) and `y` (target)
   - Splitting data into training and test sets
   - Creating and fitting a Decision Tree Regressor
   - Generating predictions
   - Evaluating performance using MAE
   - Exploring model predictions

4. **Practice and Reflection**
   - Reviewing model performance
   - Thinking about model complexity
   - Interpreting the results

## Tools and Libraries

- Python
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- Google Colab / Jupyter Notebook

## Practical Activities

The notebook is designed to be followed step by step. Read each explanation, run the related code cell, inspect the output, and complete the exercises included in the notebook.

## How to Use This Notebook

1. Click the **Open in Google Colab** badge at the top of this page.
2. Open the notebook in Google Colab.
3. Run the cells in order, from top to bottom.
4. Read the explanations and review the outputs.
5. Complete the practical exercises before moving to the next section.

> **Important:** Keep the training and test data separate. The test data is used to evaluate the model on examples it did not learn from during training.

---

**Data Science & AI Program — Machine Learning**
