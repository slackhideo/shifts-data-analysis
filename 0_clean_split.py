# Author: Tiago M. de Barros
# Date:   2022-08-26

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# File names to use
FILE_INPUT = "data_sample.xlsx"
FILE_TRAIN = "data_train.csv"
FILE_TEST  = "data_test.csv"

# Pandas display option
pd.set_option("display.max_columns", None)

# Set seaborn theme
sns.set_theme()


def main():
    "Main function"

    data = pd.read_excel(FILE_INPUT, header=0, engine="openpyxl")

    # Check data shape
    print(f"Input data shape: {data.shape}")

    # Check for missing data
    print(f"Null entries: {data.isnull().sum().sum()}\n")

    # Check data summary information
    data.info()

    # Check data descriptive statistics
    print("\n", data.describe())

    # Print negative values of time
    negative = data["Dump spot time/s"][data["Dump spot time/s"] < 0]
    print(f"\nNegative time:\n{negative}")

    # Drop these entries
    data = data.drop(negative.index)

    # Drop entries containing outliers
    data = remove_outliers(data, "Production/bcm",          140000)
    data = remove_outliers(data, "Cycle distance empty/m",   10000)
    data = remove_outliers(data, "Cycle distance full/m",    10000)
    data = remove_outliers(data, "Spot total duration/s",      120)
    data = remove_outliers(data, "Load time/s",                240)
    data = remove_outliers(data, "Dump spot time/s",            40)
    data = remove_outliers(data, "Dump spot time/s",             5, True)
    data = remove_outliers(data, "Off circuit travel time/s", 4000)
    data = remove_outliers(data, "Truck production hours",     400, True)
    data = remove_outliers(data, "Loader production hours",    100, True)

    # Check correlation between features
    sns.heatmap(data.corr(), cmap=plt.cm.Reds, annot=True, fmt=".1g")
    plt.show()

    # Drop features presenting multicollinearity
    data = data.drop(columns=["Travel empty speed/kmh",
                              "Cycle distance empty/m",
                              "Cycle distance full/m",
                              "Rise distance empty/m",
                              "Travel empty duration/s",
                              "Total cycle time/s",
                              "Number of loading units",
                              "Truck available hours",
                              "Loader available hours"])

    # Plot new correlation between features
    sns.heatmap(data.corr(), cmap=plt.cm.Reds, annot=True)
    plt.show()

    # Check multi-variate relationship (it may take some time to plot)
    sns.pairplot(data, diag_kind="kde")
    plt.show()

    print(f"\nNew data shape: {data.shape}")

    # Split data into training and test sets
    X_train, X_test = train_test_split(data,
                                       test_size=0.1,
                                       shuffle=True,
                                       random_state=42)

    # Save the sets as CSV files
    X_train.to_csv(FILE_TRAIN, header=True, index=True)
    X_test.to_csv(FILE_TEST, header=True, index=True)

    print(f"\nSaved training ({FILE_TRAIN}) and test ({FILE_TEST}) sets.")


def remove_outliers(data:      pd.DataFrame,
                    field:     str,
                    threshold: int,
                    lessthan:  bool = False) -> pd.DataFrame:
    "Removes outliers from data"

    sns.boxplot(x=data[field])
    plt.show()
    if lessthan:
        outliers = data[field][data[field] < threshold]
    else:
        outliers = data[field][data[field] > threshold]

    return data.drop(outliers.index)


if __name__ == "__main__":
    main()
