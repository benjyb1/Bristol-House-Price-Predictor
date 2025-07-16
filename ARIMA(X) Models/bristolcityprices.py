import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
tf.random.set_seed(0)
import os
import pandas as pd
import numpy as np
np.random.seed(0)

file_path = r"C:\Users\kitcr\Downloads\Bristol extracted house price index 1995-2024.xlsx"
df = pd.read_excel(file_path)

print(df.head())

# Ensure that the 'Date' column is read correctly
df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M:%S')


# Select features (temperature, humidity, and pressure as an example)
# Apply forward fill and keep it as a DataFrame
features = df[['AveragePrice', '1m%Change', '12m%Change', 'SalesVolume', 'DetachedPrice', 'SemiDetachedPrice']].fillna(0)

print(features)


# Scale the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(features)
features_scaled = scaler.transform(features)

# Define lookback and forecast window
n_lookback = 260
n_forecast = 60

# Prepare the dataset for LSTM
X = []
y = []

for i in range(n_lookback, len(features_scaled) - n_forecast + 1):
    X.append(features_scaled[i - n_lookback: i])  # Lookback period
    y.append(features_scaled[i: i + n_forecast, 0])  # Predict temperature (column index 0)

X = np.array(X)
y = np.array(y)


model1 = Sequential()
model1.add(InputLayer((n_lookback, X.shape[2])))
model1.add(LSTM(128, return_sequences=True))
model1.add(Dropout(0.2))
model1.add(LSTM(64))
model1.add(Dense(32, activation='relu'))
model1.add(Dense(n_forecast, activation='linear'))

model1.compile(loss=Huber(), optimizer=Adam(learning_rate=0.00045), metrics=[RootMeanSquaredError()])
model1.fit(X, y, epochs=50, batch_size=4, verbose=1)

# Define actual_data from the 'AveragePrice' column of the DataFrame
actual_data = df['AveragePrice'].values.flatten()  # Convert to a 1D array for easier manipulation

# Prepare the input for forecasting
X_ = features_scaled[-n_lookback:]
X_ = X_.reshape(1, n_lookback, X.shape[2])

# Predict and reshape forecasted data
Y_ = model1.predict(X_).reshape(-1, 1)

# Create a scaled version of all features, replacing the target column with predictions
scaled_forecast = features_scaled[-1:].repeat(n_forecast, axis=0)  # Repeat the last row of scaled features
scaled_forecast[:, 0] = Y_.flatten()  # Replace target column with predictions

# Inverse transform the entire forecast dataset
Y_inverse = scaler.inverse_transform(scaled_forecast)[:, 0]  # Extract the target column after inverse transformation

# Adjust the forecast to start from the last actual value
last_actual = actual_data[-1]  # Get the last value of the actual data
forecast_offset = last_actual - Y_inverse[0]  # Calculate the offset to align the forecast
Y_adjusted = Y_inverse + forecast_offset  # Adjust the forecast


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
# Smooth the forecast
Y_smoothed = moving_average(Y_adjusted, window_size=5)

padding_size = len(Y_adjusted) - len(Y_smoothed)
Y_smoothed = np.pad(Y_smoothed, (padding_size, 0), mode='edge')

# Generate forecasted dates
actual_dates = df['Date']  # Use the Date column from the DataFrame for actual dates
forecasted_dates = pd.date_range(actual_dates.iloc[-1], periods=n_forecast + 1, freq='M')[1:]  # Generate forecasted dates

# Plot the actual and forecasted data
plt.figure(figsize=(12, 6))
plt.plot(actual_dates, actual_data, label='Actual House Prices', color='blue')
plt.plot(forecasted_dates, Y_smoothed, label='Forecasted House Prices', color='red')
plt.xlabel('Date')
plt.ylabel('Average House Price (£)')
plt.legend()
plt.title('Forecast of Average House Price in Bristol')
plt.xticks(rotation=45)  # Rotate the date labels for better readability
plt.tight_layout()
plt.show()
