import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import root_mean_squared_log_error

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV

import os
import joblib

import time
import logging

# Set up logging
logging.basicConfig(
    filename="./logs/grid_search_catboost.txt",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

fold = 0

# Import the data
df = pd.read_csv("./input/train_folds.csv")

# Split the data into training and testing
train = df[df.kfold != fold].reset_index(drop=True)
test = df[df.kfold == fold].reset_index(drop=True)

# Split the data into features and target
X_train = train.drop(['id', 'Rings', 'kfold'], axis=1)
X_test = test.drop(['id', 'Rings', 'kfold'], axis=1)

y_train = train.Rings.values
y_test = test.Rings.values

# Define categorical and numerical columns
categorical_cols = ['Sex']
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]

# Create a column transformer for one-hot encoding and standard scaling
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)

# Define the base model
catboost_model = CatBoostRegressor(
    verbose=0,
    random_state=42
)

# Define the hyperparameters grid for RandomizedSearchCV
param_grid = {
    'learning_rate': [0.08, 0.085, 0.09],
    'depth': [6, 7, 8],
    'iterations': [800, 900, 1000],
    'l2_leaf_reg': [6.5, 7.5, 8.5],
    'bagging_temperature': [0.6, 0.65, 0.7],
    'border_count': [160, 192, 224],
    'random_strength': [0.25, 0.275, 0.3],
}

# Define the RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=catboost_model,
    param_distributions=param_grid,
    n_iter=100,
    scoring='neg_mean_squared_log_error',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Create a pipeline with the preprocessor and the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('random_search', random_search)
])


start = time.time()

# logging.info(f"Fold={fold}, Model={model}")

# Fit the model
pipeline.fit(X_train, y_train)

# make predictions
preds = pipeline.predict(X_test)

# Clip predictions to avoid negative values
preds = np.clip(preds, 0, None)

end = time.time()
time_taken = end - start

logging.info(f"==========================================")
logging.info(f"===== Starting training for model:  ======")
logging.info(f"==========================================")

# Calculate the R2 score
rmsle = root_mean_squared_log_error(y_test, preds)
print(f"Fold={fold}, RMSLE Score={rmsle:.4f}, Time={time_taken:.2f}sec")
logging.info(f"Fold={fold}, RMSLE Score={rmsle:.4f}, Time Taken={time_taken:.2f}sec")

print(f"Best parameters: {pipeline.named_steps['random_search'].best_params_}")
print(f"Best score: {pipeline.named_steps['random_search'].best_score_}")

logging.info("---------------** Best **--------------------")
logging.info(f"Best parameters: {random_search.best_params_}")
logging.info(f"Best score: {random_search.best_score_}")
rmsle_best = (-random_search.best_score_)**0.5
logging.info(f"Best RMSLE: {rmsle_best}")
logging.info("---------------** End **--------------------")
# Log all tested parameters and their corresponding scores
results = random_search.cv_results_
for i in range(len(results['params'])):
    logging.info(f"Params: \n{results['params'][i]}, RMSLE: {(-results['mean_test_score'][i])**0.5:.4f}")

logging.info("---------------xxxxxxxxxxxxx----------------")

# Save the best model
best_model = random_search.best_estimator_
joblib.dump(best_model, os.path.join("./models/", f"catboost_fold_{fold}.bin"))

