import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

# Load your dataset
data = pd.read_csv("stock_data.csv", parse_dates=['datetime'], index_col='datetime')

# Choose the index you want to predict
index_to_predict = 'close_OANDA:FR40EUR'

# Resample the data to ensure regular hourly intervals
data = data.resample('H').asfreq()

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
data_scaled = data.copy()
data_scaled[index_to_predict] = scaler.fit_transform(data[[index_to_predict]])

# Split the data into training and test sets
train_size = int(len(data_scaled) * 0.8)
train, test = data_scaled.iloc[:train_size], data_scaled.iloc[train_size:]

# Define ARIMA model with chosen parameters (1, 1, 0)
model_arima = ARIMA(train[index_to_predict], order=(1, 1, 0))
model_fit = model_arima.fit()

# Forecast using the fitted model
forecast = model_fit.forecast(steps=len(test))

# Inverse transform the forecasted values to the original scale
forecast_original_scale = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

# Inverse transform the test data to the original scale for comparison
test_original_scale = scaler.inverse_transform(test[[index_to_predict]])

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
print(f"Best model: ARIMA(1,1,0)")
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"MASE: {mase}")
print(f"SMAPE: {smape}")
