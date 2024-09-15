# S4E4 Regression with an Abalone Dataset

This project is part of the **30 Kaggle Challenges in 30 Days** series. It focuses on building a regression model to predict the age of abalones based on physical measurements using a synthetic dataset derived from the UCI Abalone dataset.

## Overview

The main goal of this challenge is to predict the **age** of abalones using physical features like **Length**, **Diameter**, **Whole Weight**, and others. The target variable `Rings` represents the number of rings in the abalone shell, which can be used to calculate the age by adding 1.5. The evaluation metric for this competition is **Root Mean Squared Logarithmic Error (RMSLE)**.

## Problem Description

The problem is based on the **Kaggle Playground Series Season 4, Episode 4**. The dataset has been generated synthetically, using the **Abalone dataset** from the **UC Irvine Machine Learning Repository**.

- **Kaggle Playground**: [Link to Kaggle](https://www.kaggle.com/competitions/playground-series-s4e4)
- **Original Dataset**: [UCI Abalone Dataset](https://archive.ics.uci.edu/dataset/1/abalone)


## How to Run the Project

1. **Create Folds:**
   To create stratified k-fold splits of the training data, run:

   ```bash
   python src/create_fold.py
   ```

   This will generate a new CSV file (`train_folds.csv`) with fold assignments.

2. **Train the Model:**
   You can train models by specifying the model name through the command line. For example, to run CatBoost:

   ```bash
   python src/main.py --model catboost
   ```

   Available models are defined in `model_dispatcher.py`.

3. **Grid Search for Hyperparameter Tuning:**
   To perform grid search for hyperparameter tuning on CatBoost, you can run:

   ```bash
   python src/cat_grid.py
   ```

   This will log the best hyperparameters and results in `grid_search_catboost.txt`.

## Models Used

The following models are supported for this regression problem, defined in the `model_dispatcher.py` file:

- Linear Regression
- Random Forest Regressor
- XGBoost
- LightGBM
- CatBoost
- Gradient Boosting
- SVR
- KNeighbors Regressor
- Ridge Regression
- Lasso Regression

The best-performing model during initial tests was **CatBoost**, achieving an average **RMSLE** of approximately 0.1493 across five folds.

## Notebook and Blog

- **Kaggle Notebook**: [Regression with Abalone Dataset](https://www.kaggle.com/code/surajwate/s4e4-abalone-catboost)
- **Blog Post**: [Regression with Abalone Dataset - Blog](https://surajwate.com/blog/regression-with-an-abalone-dataset/)

## Results

- **Final RMSLE Score**: 0.14783
- **Kaggle Rank Range**: 1064-1069

The CatBoost model showed the best performance with minimal hyperparameter tuning. Further improvements may come from feature engineering and ensembling models in future iterations.

## Next Steps

- **Feature Engineering**: Apply transformations, feature selection, or interactions between features.
- **Ensembling**: Combine multiple models like CatBoost, LightGBM, and XGBoost to improve performance.

