import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_excel("data_sample.xlsx", header=0, engine="openpyxl")

# Check for missing data
print(data.isnull().sum().sum())

minmax = (data - data.min()) / (data.max() - data.min())
mean = (data - data.mean()) / data.std()
