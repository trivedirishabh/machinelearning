# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# Load the dataset

dataset = pd.read_csv('titanic.csv')
# Identify the categorical data

X = dataset.iloc[:,:-1].values
print(X)
y = dataset.iloc[:,1].values
#print(y)
# Implement an instance of the ColumnTransformer class
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[8])],remainder='passthrough')
# X = np.array(ct.fit_transform(X))
# print(x)
# Apply the fit_transform method on the instance of ColumnTransformer


# Convert the output into a NumPy array


# Use LabelEncoder to encode binary categorical data


# Print the updated matrix of features and the dependent variable vector

