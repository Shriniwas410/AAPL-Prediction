#AAPL Stock Price Prediction

This repository contains a Python script for predicting the stock prices of Apple Inc. using different regression models.

#Description

The script uses historical stock price data of Apple Inc. downloaded from Yahoo! Finance in CSV format. It applies three different regression models - Linear Regression, Ridge Regression, and Lasso Regression - to predict future stock prices.
Features

- Data visualization of historical stock prices and moving averages.
- Calculation of return deviation.
- Prediction of future stock prices using Linear Regression, Ridge Regression, and Lasso Regression models.
- Comparison of the performance of the three models.
  
#How to Use

1. Clone this repository.
2. Download the historical stock price data of Apple Inc. from Yahoo! Finance in CSV format.
3. Update the path of the CSV file in the script.
4. Run the script.


#Requirements

- Python 3.6 or above
- Libraries: numpy, pandas, matplotlib, sklearn, datetime
  
#Code Structure

- The script first imports the necessary libraries.
- It then reads the CSV file containing the historical stock price data.
- The script visualizes the adjusted closing prices and calculates the moving average.
- It also calculates the return deviation.
- The script then creates a new column 'Prediction' in the dataframe, which is the 'Adj Close' column shifted by 'predict_days'.
- It creates the feature matrix X and the target vector y.
- The data is split into training and testing sets.
- The script defines the Linear Regression model, trains it, and calculates its score.
- It predicts the stock prices for the next 'predict_days' days and for the entire dataset.
- The script plots the actual and predicted prices.
- The same process is repeated for the Ridge and Lasso Regression models.
- Finally, the script compares the performance of the three models and prints the best performer.
