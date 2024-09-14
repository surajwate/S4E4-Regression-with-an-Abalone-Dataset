import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score

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
    X_train = train.drop(['id', 'FloodProbability', 'kfold'], axis=1)
    X_test = test.drop(['id', 'FloodProbability', 'kfold'], axis=1)

    y_train = train.FloodProbability.values
    y_test = test.FloodProbability.values

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform the training data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the model
    model = model_dispatcher.models[model]

    try:
        start = time.time()

        # logging.info(f"Fold={fold}, Model={model}")

        # Fit the model
        model.fit(X_train, y_train)

        # make predictions
        preds = model.predict(X_test)

        end = time.time()
        time_taken = end - start

        # Calculate the R2 score
        r2 = r2_score(y_test, preds)
        print(f"Fold={fold}, R2 Score={r2:.4f}, Time={time_taken:.2f}sec")
        logging.info(f"Fold={fold}, R2 Score={r2:.4f}, Time Taken={time_taken:.2f}sec")

        # Save the model
        joblib.dump(model, os.path.join(config.MODEL_OUTPUT, f"model_{fold}.bin"))
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

