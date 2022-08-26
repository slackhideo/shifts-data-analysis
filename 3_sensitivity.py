# Author: Tiago M. de Barros
# Date:   2022-08-26

import pandas as pd
from sklearn.linear_model import LinearRegression
from SALib.sample import saltelli
from SALib.analyze import sobol

# File names to use
FILE_TRAIN_LABELS = "data_train_labels.csv"

# Pandas display option
pd.set_option("display.max_columns", None)


def main():
    "Main function"

    # Read training data
    data_train = pd.read_csv(FILE_TRAIN_LABELS, header=0, index_col=0)

    # Check training data shape
    print(f"Training data shape: {data_train.shape}")

    # For sensitivity analysis, consider the good shifts in the training data
    good_shifts = data_train[data_train["Label"] == 1].drop(columns="Label")

    # Check good shifts data shape
    print(f"\nGood shifts data shape: {good_shifts.shape}")

    # Check good shifts data descriptive statistics
    print("\n", good_shifts.describe())


    # Below, there is a sensitivity analysis considering production as the
    # target feature in a regression problem. The goal is to verify which
    # features are more relevant to production.

    # Split production as the target feature
    X = good_shifts.iloc[:,1:]
    y = good_shifts.iloc[:,:1]

    # Train a regressor
    regressor = LinearRegression()
    regressor.fit(X, y)

    # Bounds of features in good shifts
    bounds = [[14.678644, 22.654539],
              [76.912558, 139.921601],
              [6.0, 16.0],
              [76.095238, 197.978182],
              [14.315068, 92.594185],
              [135.758706, 207.985455],
              [432.278794, 774.076923],
              [8.440936, 44.910156],
              [9.437148, 24.877907],
              [40.157576, 71.615205],
              [208.220322, 1858.940571],
              [31.0, 55.0],
              [0.493750, 0.702649],
              [469.104444, 597.860000],
              [112.684722, 165.088889]]

    problem = {"num_vars": 15,
               "names": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
                         "x10", "x11", "x12", "x13", "x14", "x15"],
               "bounds": bounds}

    # Generate values for the input features
    param_values = saltelli.sample(problem, 1024)

    # Get the predicted production values
    Y = regressor.predict(param_values).ravel()

    # Get the sensitivities
    Si = sobol.analyze(problem, Y)

    print(f"\nFirst-order sensitivities:\n{Si['S1']}")
    print(f"\nTotal-order sensitivities:\n{Si['ST']}")

    # From this, we see that production has the highest sensitivity with
    # relation to the number of trucks


if __name__ == "__main__":
    main()
