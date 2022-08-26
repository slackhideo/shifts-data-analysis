# Author: Tiago M. de Barros
# Date:   2022-08-26

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# File names to use
FILE_TRAIN        = "data_train.csv"
FILE_TEST         = "data_test.csv"
FILE_TRAIN_LABELS = "data_train_labels.csv"
FILE_TEST_LABELS  = "data_test_labels.csv"

# Number of clusters to consider
N_CLUSTERS = 3

# Pandas display option
pd.set_option("display.max_columns", None)

# Set seaborn theme
sns.set_theme()


def main():
    "Main function"

    # Read training data
    data_train = pd.read_csv(FILE_TRAIN, header=0, index_col=0)

    # Check training data shape
    print(f"Training data shape: {data_train.shape}")

    # Scale the data per feature
    scaler = StandardScaler()
    scaled_data_train = scaler.fit_transform(data_train)

    # Principal Component Analysis (PCA)
    pca = PCA(random_state=42)
    pca.fit(scaled_data_train)
    print("\nExplained variance ratio using Principal Component Analysis:\n"
            f"{pca.explained_variance_ratio_}")
    graph = sns.lineplot(data=np.cumsum(pca.explained_variance_ratio_))
    graph.set_xticks(range(len(pca.explained_variance_ratio_)))
    graph.set_xticklabels(range(1, len(pca.explained_variance_ratio_) + 1))
    graph.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    graph.set(title="Principal Component Analysis",
              xlabel="Number of components",
              ylabel="Explained variance")
    plt.show()

    # Clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    kmeans.fit(scaled_data_train)

    # Coordinates of cluster centers in original scaling
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # Get the thresholds for production value
    print("\nProduction coordinates of cluster centers:")
    for cluster, coordinate in enumerate(cluster_centers[:, 0]):
        print(cluster, coordinate, sep=": ")
    production_coord = np.sort(cluster_centers[:, 0])
    production_thresholds = [np.mean(production_coord[i:(i + 2)])
            for i in range(len(production_coord) - 1)]
    print(f"\nThresholds for production: {production_thresholds}")


    # Read test data
    data_test = pd.read_csv(FILE_TEST, header=0, index_col=0)

    # Check test data shape
    print(f"\nTest data shape: {data_test.shape}")

    scaled_data_test = scaler.transform(data_test)
    predicted_clusters = kmeans.predict(scaled_data_test)
    print(f"\nPredicted clusters:\n{predicted_clusters}")


    # Assign classification labels according to clusters
    # 1: good shift
    # 0: average shift
    #-1: bad shift
    labels_train = [1 if label == 0 else (0 if label == 2 else -1)
            for label in kmeans.labels_]
    labels_test  = [1 if label == 0 else (0 if label == 2 else -1)
            for label in predicted_clusters]

    data_train["Label"] = labels_train
    data_test["Label"]  = labels_test

    # Distribution of shifts per label
    labels, counts = np.unique(labels_train, return_counts=True)
    print("\nDistribution of shifts in training data")
    for label, count in zip(labels, counts):
        print(f"{label:2}: {count:3}")
    labels, counts = np.unique(labels_test, return_counts=True)
    print("\nDistribution of shifts in test data")
    for label, count in zip(labels, counts):
        print(f"{label:2}: {count:3}")

    # Save the sets with labels as CSV files
    data_train.to_csv(FILE_TRAIN_LABELS, header=True, index=True)
    data_test.to_csv(FILE_TEST_LABELS, header=True, index=True)

    print(f"\nSaved training ({FILE_TRAIN_LABELS}) "
            f"and test ({FILE_TEST_LABELS}) sets with labels.")


if __name__ == "__main__":
    main()
