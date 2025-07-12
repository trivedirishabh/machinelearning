# y = b+b1x
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
#print(dataset)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

#splitting the dataset into training & test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)

#training the model on the training set

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

# creating visualization for training set
# plt.scatter(X_train, y_train, color= 'red')
# plt.plot(X_train, regressor.predict(X_train), color="blue")
# plt.title("Salary Vs Experience (Training set)")
# plt.xlabel("years of Experience")
# plt.ylabel("salary")
# #plt.show()

# Creating visualization for Test set
plt.scatter(X_test, y_test, color= 'red')
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary Vs Experience (Test set)")
plt.xlabel("years of Experience")
plt.ylabel("salary")
plt.show()


