import pandas as pd


stock_data_path = 'stock_data.csv'
news_data_path = 'reuters_news_scores.csv'

stock_data = pd.read_csv(stock_data_path)
news_data = pd.read_csv(news_data_path)

# Convert timestamp and datetime to proper datetime formats
news_data['timestamp'] = pd.to_datetime(news_data['timestamp'], format='%m/%d/%Y %H:%M')
stock_data['datetime'] = pd.to_datetime(stock_data['datetime'], format='%Y-%m-%d %H:%M:%S')

# Compute the final score
news_data['final_score'] = news_data['relevance'] * news_data['sentiment']

# Set timestamp as index in news_data
news_data.set_index('timestamp', inplace=True)

# Calculate the rolling sentiment (7 days rolling sum of final_score)
rolling_sentiment = news_data['final_score'].rolling('7d').sum().reset_index()

# Rename the column to 'rolling_sentiment'
rolling_sentiment.rename(columns={'final_score': 'rolling_sentiment'}, inplace=True)

# Merge rolling sentiment with stock data using datetime and timestamp
merged_data = pd.merge_asof(stock_data.sort_values(by='datetime'),
                            rolling_sentiment.sort_values(by='timestamp'),
                            left_on='datetime', right_on='timestamp', direction='backward')

# Drop unnecessary columns (like timestamp from news)
merged_data.drop(columns=['timestamp'], inplace=True)

# Save the merged data to a new CSV file (optional)
merged_data.to_csv('rolling_sentiment.csv', index=False)

# Display the first few rows of the merged data
print(merged_data.head())
