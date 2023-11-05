"""AAPL Stock Price Prediction.ipynb

This script predicts the stock prices of Apple Inc. using different regression models.
"""

# Importing all the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Reading the CSV file
df = pd.read_csv('/AAPL.csv')

# Setting 'Date' as the index of the dataframe
df.set_index('Date', inplace=True)

# Plotting the adjusted closing prices
df['Adj Close'].plot(label='AAPL', figsize=(15, 9), title='Adjusted Closing Price', color='red', linewidth=1.0, grid=True)
plt.legend()

# Calculating the moving average with a window size of 100
mvag = df['Adj Close'].rolling(window=100).mean()

# Plotting the adjusted closing prices and moving average
df['Adj Close'].plot(label='AAPL', figsize=(15,10), title='Adjusted Closing Price vs Moving Average', color='red', linewidth=1.0, grid=True)
mvag.plot(label='MVAG', color='blue')
plt.legend()

# Calculating the return deviation
rd = df['Adj Close'] / df['Adj Close'].shift(1) - 1
rd.plot(label='Return', figsize=(15, 10), title='Return Deviation', color='red', linewidth=1.0, grid=True)
plt.legend()

# Number of days for which to predict the stock prices
predict_days = 30

# Creating a new column 'Prediction' shifted by 'predict_days'
df['Prediction'] = df['Adj Close'].shift(-predict_days)

# Creating the feature matrix X by dropping the 'Prediction' column
X = np.array(df.drop(['Prediction'], axis = 1))
X = X[:-predict_days]

# Creating the target vector y
y = np.array(df['Prediction'])
y = y[:-predict_days]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Defining the Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Calculating the score of the Linear Regression Model
linear_model_score = linear_model.score(X_test, y_test)

# Predicting the stock prices for the next 'predict_days' days
X_predict = np.array(df.drop(['Prediction'], 1))[-predict_days:]
linear_model_predict_prediction = linear_model.predict(X_predict)

# Predicting the stock prices for the entire dataset
linear_model_real_prediction = linear_model.predict(np.array(df.drop(['Prediction'], 1)))

# Plotting the actual and predicted prices
plt.figure(figsize=(15, 9))
plt.plot(df.index, linear_model_real_prediction, label='Linear Prediction', color='blue')
plt.plot(df.index[-predict_days:], linear_model_predict_prediction, label='Forecast', color='green')
plt.plot(df.index, df['Close'], label='Actual', color='red')
plt.legend()

# Repeating the same process for Ridge and Lasso Regression models

# Ridge Regression
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)
ridge_model_score = ridge_model.score(X_test, y_test)
ridge_model_predict_prediction = ridge_model.predict(X_predict)
ridge_model_real_prediction = ridge_model.predict(np.array(df.drop(['Prediction'], 1)))

# Lasso Regression
lasso_model = Lasso()
lasso_model.fit(X_train, y_train)
lasso_model_score = lasso_model.score(X_test, y_test)
lasso_model_predict_prediction = lasso_model.predict(X_predict)
lasso_model_real_prediction = lasso_model.predict(np.array(df.drop(['Prediction'], 1)))

# Comparing the performance of the three models
best_score = max(linear_model_score, ridge_model_score, lasso_model_score)
index = np.argmax([linear_model_score, ridge_model_score, lasso_model_score])
best_regressor = {0:'Linear Regression Model', 1:'Ridge Model', 2:'Lasso Model'}
print("The Best Performer is {0} with the score of {1}%.".format(best_regressor[index], best_score*100))

