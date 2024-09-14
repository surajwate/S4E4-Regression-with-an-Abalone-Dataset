import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import root_mean_squared_log_error

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import os
import argparse
import joblib

import config
import model_dispatcher

import time
import logging

# Set up logging
logging.basicConfig(
    filename=config.LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

def run(fold, model):
    # Import the data
    df = pd.read_csv(config.TRAINING_FILE)

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

    # Create a pipeline with the preprocessor and the model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model_dispatcher.models[model])
    ])


    try:
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

        # Calculate the R2 score
        rmsle = root_mean_squared_log_error(y_test, preds)
        print(f"Fold={fold}, R2 Score={rmsle:.4f}, Time={time_taken:.2f}sec")
        logging.info(f"Fold={fold}, R2 Score={rmsle:.4f}, Time Taken={time_taken:.2f}sec")

        # Save the model
        joblib.dump(pipeline, os.path.join(config.MODEL_OUTPUT, f"model_{fold}.bin"))
    except Exception as e:
        logging.exception(f"Error occurred for Fold={fold}, Model={model}: {str(e)}")
    

if __name__ == '__main__':
    # Initialize the ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # Add the arguments to the parser
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)

    # Read the arguments from the command line
    args = parser.parse_args()

    # Run the fold specified by the command line arguments
    run(fold=args.fold, model=args.model)

