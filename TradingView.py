import sys
import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import warnings
 
# pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git
from tvDatafeed import TvDatafeed, Interval

#---------------------------------------------------------------------------------

# Disable warnings
warnings.filterwarnings ('ignore')

#---------------------------------------------------------------------------------

# Function to fetch data from TradingView
def get_data_tv(symbols):
    tv = TvDatafeed()
    
    # Function to fetch data and rename columns
    def fetch_and_rename(symbol):
        print("Fetching: " + symbol)
        exchange, symbol_code = symbol.split(':')
        # Fetch a larger number of bars to get more historical data
        data = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_hour, n_bars=5000, fut_contract=1)
        data = data.reset_index()
        data.columns = ['datetime'] + data.columns[1:].map(lambda x: x + f'_{symbol}').tolist()
        return data
    
    # Fetch and process data for each symbol
    data_frames = [fetch_and_rename(symbol) for symbol in symbols]
    
    # Merge all DataFrames on 'datetime' column using reduce
    merged_data = reduce(lambda left, right: pd.merge(left, right, on='datetime'), data_frames)

    # Filter the data between June 11 and August 31
    merged_data['datetime'] = pd.to_datetime(merged_data['datetime'])
    filtered_data = merged_data[(merged_data['datetime'] >= '2024-06-11') & (merged_data['datetime'] <= '2024-08-31')]
    
    # Normalize the close columns for all symbols
    for symbol in symbols:
        close_column = f'close_{symbol}'
        filtered_data[close_column] = filtered_data[close_column] / filtered_data[close_column].iloc[0]

    return filtered_data

#---------------------------------------------------------------------------------

# Get data
symbols = ["CAPITALCOM:DE40", "CAPITALCOM:UK100", "CAPITALCOM:SP35", "CAPITALCOM:US500", "OANDA:FR40EUR"]
data = get_data_tv(symbols)

# Print the filtered data
print(data.head())

# Save the filtered data to CSV
data.to_csv('stock_data.csv', index=False)
print("Filtered data has been saved to 'stock_data.csv'.")

# Plot filtered data
plt.figure(figsize=(10, 6))

plt.plot(data['datetime'], data['close_CAPITALCOM:DE40'], linestyle='-', label='DE40')
plt.plot(data['datetime'], data['close_CAPITALCOM:UK100'], linestyle='-', label='UK100')
plt.plot(data['datetime'], data['close_CAPITALCOM:US500'], linestyle='-', label='US500')
plt.plot(data['datetime'], data['close_OANDA:FR40EUR'], linestyle='-', label='FR40')

plt.title('Time Series Plot (Filtered)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()
