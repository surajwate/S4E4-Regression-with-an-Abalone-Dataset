import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def create_fold(fold):
    # import data
    data = pd.read_csv('./input/train.csv')

    data['kfold'] = -1

    data = data.sample(frac=1).reset_index(drop=True)

    num_bins = int(np.floor(1 + np.log2(len(data))))
    data["bins"] = pd.cut(data["FloodProbability"], bins=num_bins, labels=False)

    y = data["bins"].values

    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)

    for f, (t_, v_) in enumerate(skf.split(X=data, y=y)):
        data.loc[v_, 'kfold'] = f

    data = data.drop("bins", axis=1)

    data.to_csv('./input/train_folds.csv', index=False)

if __name__ == '__main__':
    create_fold(5)