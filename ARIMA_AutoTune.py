import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima
from scipy import stats
 
 
# Function to calculate SMAPE
def smape(actual, forecast):
    denominator = (np.abs(actual) + np.abs(forecast)) / 2.0
    diff = np.abs(actual - forecast) / denominator
    return np.mean(diff) * 100

# Load your dataset
data = pd.read_csv("stock_data.csv", parse_dates=['datetime'], index_col='datetime')

# Choose the index you want to predict (can be changed)
index_to_predict = 'close_CAPITALCOM:DE40'

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

# Auto ARIMA to search a wider range for p, d, q parameters without stepwise search
model_auto_arima = auto_arima(train[index_to_predict],
                              seasonal=False,  # No seasonality for ARIMA
                              trace=True,  # Enable output
                              error_action='ignore',  # Ignore non-fatal errors
                              suppress_warnings=True,  # Ignore warnings
                              stepwise=False,  # Disable stepwise search
                              max_p=10, max_q=10,  # Test up to AR=10, MA=10
                              max_d=3,  # Differencing up to 3
                              maxiter=500)  # Increase iterations for better convergence

# Summary of the auto ARIMA model
print(model_auto_arima.summary())

# Fit the model
model_auto_arima.fit(train[index_to_predict])

# Forecast using the fitted model
forecast_auto_arima = model_auto_arima.predict(n_periods=len(test))

# Convert the forecast to a numpy array (from Series) before inverse transforming
forecast_auto_arima = np.array(forecast_auto_arima).reshape(-1, 1)

# Inverse transform the forecasted values to the original scale
forecast_original_scale = scaler.inverse_transform(forecast_auto_arima)

# Inverse transform the test data to the original scale for comparison
test_original_scale = scaler.inverse_transform(test[[index_to_predict]])

# Calculate error metrics
mse = mean_squared_error(test_original_scale, forecast_original_scale)
mae = mean_absolute_error(test_original_scale, forecast_original_scale)
mase = mae / mean_absolute_error(train[index_to_predict][1:], train[index_to_predict][:-1])
smape_value = smape(test_original_scale, forecast_original_scale)

# Print the model's error metrics
print(f"MSE: {mse}, MAE: {mae}, MASE: {mase}, SMAPE: {smape_value}")
