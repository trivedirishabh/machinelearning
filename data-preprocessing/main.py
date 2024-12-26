#pandas to convert data into dataframes
import pandas as pd
#numpy to perform mathematical operations
import numpy as np
#matplotlib to plot graphs
import matplotlib.pyplot as plt
#scikit learn for dealing with missing data, encoding
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("Data.csv")

# creating matrix of features and dependent variable
## features are used to predict the dependent variable

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

#taking care of missing data
## use scikit learn library to replace missing data with mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# One hot encoding to encode catgorical non-binary values
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])],remainder='passthrough')

X = np.array(ct.fit_transform(X))
#print(X)

# For binary values such as yes/no, we can use label encoding

le = LabelEncoder()
y = le.fit_transform(y)

# print(y)

#Splitting dataset into training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=1)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)
# apply feature scaling on training and test data
### standardization or Normalization is used to do feature scaling
###standarization gives value -1 to +1
####normalization gives values from 0 to 1

sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
print(X_train)
X_test[:,3:] = sc.transform(X_test[:,3:])
print(X_test)