import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

# Load your dataset
data = pd.read_csv("rolling_sentiment.csv", parse_dates=['datetime'], index_col='datetime')

# Choose the index and sentiment columns
index_to_predict = 'close_OANDA:FR40EUR'
sentiment_column = 'rolling_sentiment'

# Resample the data to ensure regular hourly intervals
data = data.resample('H').asfreq()

# Fill missing values
data.fillna(method='ffill', inplace=True)  # Forward fill for stock data
data[sentiment_column].fillna(method='ffill', inplace=True)  # Forward fill for sentiment

# Check for any remaining NaN values
if data[sentiment_column].isna().sum() > 0:
    print("There are still missing values in the sentiment column. Applying backward fill.")
    data[sentiment_column].fillna(method='bfill', inplace=True)

# Scale the data using MinMaxScaler
scaler_stock = MinMaxScaler()
scaler_sentiment = MinMaxScaler()

# Scale both stock price and sentiment
data_scaled = data.copy()
data_scaled[index_to_predict] = scaler_stock.fit_transform(data[[index_to_predict]])
data_scaled[sentiment_column] = scaler_sentiment.fit_transform(data[[sentiment_column]])

# Split the data into training and test sets
train_size = int(len(data_scaled) * 0.8)
train, test = data_scaled.iloc[:train_size], data_scaled.iloc[train_size:]

# Ensure no NaN values in train and test sets
train = train.dropna()
test = test.dropna()

# Define ARIMA model with chosen parameters (1, 1, 0) and sentiment as exogenous variable
model_arima = ARIMA(train[index_to_predict], order=(1, 1, 0), exog=train[sentiment_column])
model_fit = model_arima.fit()

# Forecast using the fitted model with sentiment as an exogenous variable for the test set
forecast = model_fit.forecast(steps=len(test), exog=test[sentiment_column])

# Inverse transform the forecasted values to the original scale
forecast_original_scale = scaler_stock.inverse_transform(np.array(forecast).reshape(-1, 1))

# Inverse transform the test data to the original scale for comparison
test_original_scale = scaler_stock.inverse_transform(test[[index_to_predict]])

# Calculate error metrics (on the original scale)
mse = mean_squared_error(test_original_scale, forecast_original_scale)
mae = mean_absolute_error(test_original_scale, forecast_original_scale)

# Calculate MASE on the scaled data
train_naive_forecast = train[index_to_predict][1:].values
train_actual = train[index_to_predict][:-1].values
mase_naive_error = np.mean(np.abs(train_actual - train_naive_forecast))
mase = mae / mase_naive_error  # Correctly compare errors on the same scale

# Calculate SMAPE on the original scale
smape = np.mean(2 * np.abs(forecast_original_scale - test_original_scale) / (np.abs(forecast_original_scale) + np.abs(test_original_scale))) * 100

# Print the model's error metrics
print(f"Best model: ARIMA(1,1,0) with sentiment")
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"MASE: {mase}")
print(f"SMAPE: {smape}")
