# Author: Tiago M. de Barros
# Date:   2022-08-26

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest

# File names to use
FILE_TRAIN_LABELS = "data_train_labels.csv"


def main():
    "Main function"

    # Read training data
    data_train = pd.read_csv(FILE_TRAIN_LABELS, header=0, index_col=0)

    # Check training data shape
    print(f"Training data shape: {data_train.shape}")

    # Split target label from input features
    X = data_train.iloc[:, :16]
    y = data_train.iloc[:, 16]

    # Get the scores of features based on univariate statistical tests
    select = SelectKBest(k="all")
    select.fit(X, y)
    print("\nFeature scores:")
    order = np.flip(np.argsort(select.scores_))
    for column, score in zip(X.columns[order], select.scores_[order]):
        print(f"{column:25}: {score:7.3f}")


if __name__ == "__main__":
    main()
