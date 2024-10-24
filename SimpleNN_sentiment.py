import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the merged stock and sentiment data CSV file
df = pd.read_csv('rolling_sentiment.csv')

# Define the index and sentiment columns 
index_column = 'close_OANDA:FR40EUR'
sentiment_column = 'rolling_sentiment'

# Ensure there are no missing values in the selected columns
df = df.dropna(subset=[index_column, sentiment_column])

# Scale the stock prices and sentiment values to be between 0 and 1 for better NN performance
scaler_stock = MinMaxScaler()
scaler_sentiment = MinMaxScaler()

df[index_column] = scaler_stock.fit_transform(df[index_column].values.reshape(-1, 1))
df[sentiment_column] = scaler_sentiment.fit_transform(df[sentiment_column].values.reshape(-1, 1))

# Prepare the dataset for training (using a sequence length and including sentiment)
def create_sequences_with_sentiment(data, sentiment, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequence = np.column_stack((data[i:i + seq_length], sentiment[i:i + seq_length]))  # Add sentiment as feature
        sequences.append(sequence)
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

seq_length = 10  # Number of time steps to consider
data = df[index_column].values
sentiment = df[sentiment_column].values
sequences, labels = create_sequences_with_sentiment(data, sentiment, seq_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, shuffle=False)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define the Simple Neural Network with additional input for sentiment
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set the parameters
input_size = seq_length * 2  # Because we now have stock data and sentiment data (2 features per timestep)
hidden_size = 64
output_size = 1

# Initialize the model, loss function, and optimizer
model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the SimpleNN model
epochs = 400
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train.view(X_train.size(0), -1))  # Flatten input for linear layer
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test.view(X_test.size(0), -1))  # Flatten input for linear layer
    test_loss = criterion(predictions, y_test).item()
    print(f'Test Loss: {test_loss:.4f}')

# Inverse scale the predictions and the actual values
predicted_stock_price = scaler_stock.inverse_transform(predictions.numpy())
actual_stock_price = scaler_stock.inverse_transform(y_test.numpy())

# Calculate MSE, MAE, MASE, and SMAPE

def calculate_mase(y_true, y_pred, y_train):
    naive_forecast = np.mean(np.abs(np.diff(y_train, axis=0)))
    mase = np.mean(np.abs(y_true - y_pred)) / naive_forecast
    return mase

def calculate_smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

mse = mean_squared_error(actual_stock_price, predicted_stock_price)
mae = mean_absolute_error(actual_stock_price, predicted_stock_price)
mase = calculate_mase(actual_stock_price, predicted_stock_price, scaler_stock.inverse_transform(y_train.numpy()))
smape = calculate_smape(actual_stock_price, predicted_stock_price)

# Print all metrics
print(f'MSE: {mse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MASE: {mase:.4f}')
print(f'SMAPE: {smape:.4f}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(actual_stock_price, label='Actual Prices')
plt.plot(predicted_stock_price, label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Actual vs Predicted Stock Prices for Index (with Sentiment)')
plt.legend()
plt.show()