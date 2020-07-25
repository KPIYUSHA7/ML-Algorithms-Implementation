#Simple linear Regression
                    #PART 1 : Data preprocessing 

#Import the libraries
                    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('Salary_Data.csv')

#Create a MATRIX of independant features
X = dataset.iloc[:,:-1].values #All rows, all columns except last column
X

#Create dependant variable VECTOR
y = dataset.iloc[:, 1].values #Last column of dataset
y

#Splitting the dataset into training and test set - test set is 20%[RANDOM SPLIT]
from sklearn.model_selection import train_test_split

#Put 10 observations in test and 20 in training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fitting Simple Linear regression to training set
from sklearn.linear_model import LinearRegression
#Create an object
regressor = LinearRegression()
#Fit the model to training data using fit method
regressor.fit(X_train, y_train)

#Predict test set results
#Put predicted salaries in a vector y_pred
y_pred = regressor.predict(X_test)


#Visualizing the training set results
#Plot observation points
plt.scatter(X_train, y_train, color = 'red') #REAL VALUES
#Plot regression line - x_train and predicted values for x_train
plt.plot(X_train,regressor.predict(X_train), color = 'blue') #PREDICTED VALUES
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Experience(years)')
plt.ylabel('Salary ($)')
plt.show()


#Visualizing the test set results
#Plot observation points
plt.scatter(X_test, y_test, color = 'red') #REAL VALUES
#Plot regression line - same for x_train and x_test; No need to change
plt.plot(X_train,regressor.predict(X_train), color = 'blue') #PREDICTED VALUES
plt.title('Salary vs. Experience (Test Set)')
plt.xlabel('Experience(years)')
plt.ylabel('Salary ($)')
plt.show()