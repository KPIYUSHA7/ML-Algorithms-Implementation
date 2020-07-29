#Multiple Linear Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
dataset = pd.read_csv('50_Startups.csv')

#Set the independant variables - everything except profit
X = dataset.iloc[:, :-1]

#set the dependant variable - Profit
y = dataset.iloc[:, 4].values

#Encode the 'state' categorical variable
X = pd.get_dummies(X)

#Avoiding the dummy variable trap
#Remove the last column in X - Remove one dummy variable; Drops New York
X = X.iloc[:, :-1]

# Splitting the dataset into the Training set and Test set - 10 in test and 40 in training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling - Built-in in Multiple Linear Regression library

#Fit Multiple Linear Regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.api as sm #Does not take b0 constant into account
#Append column of ones to the X matrix 
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

#Build the optimal matrix containing the optimal team of statistically significant IV
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

#Look for predictor with highest p-value
regressor_OLS.summary()
'''Highest p-value is that obtained by the independant
variable X4 - 4th column in original X'''
#Remove the independant variable with highest p-value
X_opt = X[:, [0,1,2,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Remove x4 - next highest p-value of 0.94
'''NOTE - LOOK AT THE ORIGINAL MATRIX X while removing the IV'''
'''X4 - 5th column in original X'''
X_opt = X[:, [0,1,2,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Remove x2 - next highest p-value of 0.602
'''X2 - 2nd column in the original X'''
X_opt = X[:, [0,1,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#R and D spend - HIGHEST effect on profit
'''Marketing spend - 6% - for SL of 5% this is higher, but if
significance level was 10%, this would have qualified to 
be kept in the model'''
#Remove the x2 to follow B Elimination thoroughly
'''X3 - 3rd column in original X'''
X_opt = X[:, [0,1]] 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

'''R and D spend - Most powerful IV to predict profit''' 

