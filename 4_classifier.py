# Author: Tiago M. de Barros
# Date:   2022-08-26

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

# File names to use
FILE_TRAIN_LABELS = "data_train_labels.csv"
FILE_TEST_LABELS  = "data_test_labels.csv"


def main():
    "Main function"

    # Read input data
    data_input = pd.read_csv(FILE_TRAIN_LABELS, header=0, index_col=0)

    # Check input data shape
    print(f"Input      data shape: {data_input.shape}")

    # Split target label from input features
    X_input = data_input.iloc[:, :16]
    y_input = data_input.iloc[:, 16]

    # Create training and evaluation sets
    X_train, X_eval, y_train, y_eval = train_test_split(X_input,
                                                        y_input,
                                                        test_size=0.1,
                                                        shuffle=True,
                                                        random_state=42,
                                                        stratify=y_input)

    # Check training and evaluation data shape
    print(f"Training   data shape: {X_train.shape}, {y_train.shape}")
    print(f"Evaluation data shape: {X_eval.shape}, {y_eval.shape}")

    models_results = []

    print("\n##############")
    print("# Linear SVM #")
    print("##############")
    model = SVC(random_state=42, kernel="linear")
    parameters = [{"C": [0.01, 0.1, 1, 10, 100, 1000]}]
    best = cross_validation(model, parameters, X_train, X_eval, y_train, y_eval)
    score = print_results(best, X_eval, y_eval)
    models_results.append((score, best))

    print("\n#######################")
    print("# SVM with RBF kernel #")
    print("#######################")
    model = SVC(random_state=42, kernel="rbf")
    parameters = [{"C": [0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8],
                   "gamma": ["auto", "scale", 0.001, 0.01, 0.1]}]
    best = cross_validation(model, parameters, X_train, X_eval, y_train, y_eval)
    score = print_results(best, X_eval, y_eval)
    models_results.append((score, best))

    print("\n#################")
    print("# Random Forest #")
    print("#################")
    model = RandomForestClassifier(random_state=42)
    parameters = [{"n_estimators": [5, 10, 50, 100, 500, 1000],
                   "max_depth": [5, 10, None],
                   "max_features": [1, 2, "auto", "sqrt", "log2", None],
                   "class_weight": ["balanced", None]}]
    best = cross_validation(model, parameters, X_train, X_eval, y_train, y_eval)
    score = print_results(best, X_eval, y_eval)
    models_results.append((score, best))

    print("\n#####################")
    print("# Gradient Boosting #")
    print("#####################")
    model = GradientBoostingClassifier(random_state=42)
    parameters = [{"learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                   "n_estimators": [100, 200, 500],
                   "max_features": ["sqrt", "log2", None]}]
    best = cross_validation(model, parameters, X_train, X_eval, y_train, y_eval)
    score = print_results(best, X_eval, y_eval)
    models_results.append((score, best))


    # Get the best prediction model
    best_index = np.argmax(np.array(models_results, dtype=object)[:, 0])
    best_model = models_results[best_index][1]
    print(f"\nBest model: {best_model}")


    # Read test data
    data_test = pd.read_csv(FILE_TEST_LABELS, header=0, index_col=0)

    # Check test data shape
    print(f"\nTest data shape: {data_test.shape}")

    # Split target label from input features
    X_test = data_test.iloc[:, :16]
    y_test = data_test.iloc[:, 16]

    # Fit the best model
    best_model.fit(X_input, y_input)
    print("\nFinal results:")
    print_results(best_model, X_test, y_test)


def cross_validation(model, parameters, X_train, X_eval, y_train, y_eval):
    "Perform cross validation to get the best values for hyperparameters"

    clf = GridSearchCV(model, parameters, cv=10, refit=True, n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)

    print("Best result:")
    print(f"{clf.best_score_:0.3f} for {clf.best_params_}", end="\n\n")
    print("Grid scores:")
    means = clf.cv_results_["mean_test_score"]
    stds  = clf.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
        print(f"{mean:0.3f} (+/-{2 * std:0.03f}) for {params}")
    print()

    return clf.best_estimator_


def print_results(model, X_eval, y_eval):
    "Print the accuracy and F1 score of a model"

    score = model.score(X_eval, y_eval)
    print(f"Accuracy: {score:0.3f}")
    clazz = model.predict(X_eval)
    fscore = f1_score(y_eval, clazz, average="macro")
    print(f"F1 score: {fscore:0.3f}")

    return score


if __name__ == "__main__":
    main()
