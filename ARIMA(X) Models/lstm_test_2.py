import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from tensorflow.keras.regularizers import l2
from pandas.tseries.offsets import DateOffset
from tensorflow.keras.models import load_model
import os
import pandas as pd
import numpy as np


np.random.seed(42)
tf.random.set_seed(42)

file_path = r"C:\Users\kitcr\Downloads\Bristol extracted house price index 1995-2024.xlsx"
df = pd.read_excel(file_path)

print(df.head())

df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M:%S')
actual_dates = df['Date']
actual_dates = actual_dates.iloc[-18:]# Use the Date column from the DataFrame for actual dates
print(actual_dates)
features = df[['AveragePrice', 'Index', '1m%Change', '12m%Change', 'SalesVolume', 'DetachedPrice', 'DetachedIndex', 'SemiDetachedPrice']].fillna(0)

# Scale the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(features)
features_scaled = scaler.transform(features)

# Define lookback and forecast window
n_lookback = 12
n_forecast = 1

# Prepare the dataset for LSTM
X = []
y = []

for i in range(n_lookback, len(features_scaled) - n_forecast + 1):
    X.append(features_scaled[i - n_lookback: i])  # Lookback period
    y.append(features_scaled[i: i + n_forecast, 0])

X = np.array(X)
y = np.array(y)

# Step 1: Cross-validation with TimeSeriesSplit
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)
training_losses = []
test_losses = []
print("Cross-Validation Results:")
for fold, (train_index, test_index) in enumerate(tscv.split(X), start=1):
    print(f"\nFold {fold}:")

    # Split data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Build the LSTM model
    model = Sequential()
    model.add(InputLayer((n_lookback, X.shape[2])))
    model.add(LSTM(64))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(n_forecast, activation='linear'))

    model.compile(loss=Huber(), optimizer=Adam(learning_rate=0.00047), metrics=[RootMeanSquaredError()])
    # Train the model and log training loss
    history = model.fit(X_train, y_train, epochs=300, batch_size=10, verbose=1)
    training_loss = history.history['loss'][-1]  # Last training loss value
    training_losses.append(training_loss)
    model.save('model')

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    test_loss = mean_squared_error(y_test.flatten(), y_pred.flatten())
    test_losses.append(test_loss)

    print(f"  Training Loss: {training_loss:.4f}")
    print(f"  Test Loss (MSE): {test_loss:.4f}")


    def simple_moving_average(values, window):
        """Compute Simple Moving Average."""
        return np.convolve(values, np.ones(window) / window, mode='valid')
    # Apply to LSTM predictions
    y_pred_smooth = simple_moving_average(y_pred.flatten(), window=5)


    plt.figure(figsize=(10, 6))
    plt.plot(y_test.flatten(), label='Actual')
    plt.plot(y_pred_smooth, label="Smoothed Predictions", linestyle="--")
    plt.title('Actual vs. Predicted Values')
    plt.legend()
    plt.show()

# Print average training and test losses
avg_training_loss = np.mean(training_losses)
avg_test_loss = np.mean(test_losses)
print(f"\nAverage Training Loss: {avg_training_loss:.4f}")
print(f"Average Test Loss: {avg_test_loss:.4f}")

# Step 2: Train the model on the full dataset and forecast future values
print("\nTraining on the full dataset for final forecasting...")
# Build the LSTM model
model1 = Sequential()
model1.add(InputLayer((n_lookback, X.shape[2])))
model1.add(LSTM(64))
model1.add(Dense(8, activation='relu'))
model1.add(Dense(n_forecast, activation='linear'))

model1.compile(loss=Huber(), optimizer=Adam(learning_rate=0.0005), metrics=[RootMeanSquaredError()])
cp1 = ModelCheckpoint('model1/', save_best_only=True)

model = load_model('model')
model.fit(X, y, epochs=300, batch_size=10, verbose=1, shuffle=False)

# Define actual_data from the 'AveragePrice' column of the DataFrame
actual_data = df['AveragePrice'].values.flatten()  # Convert to a 1D array for easier manipulation

# Prepare the input for forecasting
X_ = features_scaled[-n_lookback:]
X_ = X_.reshape(1, n_lookback, X.shape[2])

# Predict and reshape forecasted data
Y_ = model.predict(X_).reshape(-1, 1)

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
Y_smoothed = moving_average(Y_adjusted, window_size=1)
padding_size = len(Y_adjusted) - len(Y_smoothed)

Y_smoothed = np.pad(Y_smoothed, (padding_size, 0), mode='edge')

# Generate forecasted dates
actual_dates = df['Date']
actual_dates = actual_dates.iloc[-12:]# Use the Date column from the DataFrame for actual dates
#forecasted_dates = pd.date_range(actual_dates.iloc[-1], periods=n_forecast + 1, freq='M')[1:]  # Generate forecasted dates
forecasted_dates = [actual_dates.iloc[-2] + DateOffset(months=i) for i in range(1, n_forecast + 1)]
# Plot the actual and forecasted data
plt.figure(figsize=(12, 6))
plt.plot(actual_dates, actual_data[-12:], label='Actual House Prices', color='blue')
plt.plot(forecasted_dates, Y_smoothed, label='Forecasted House Prices', color='red')
plt.xlabel('Date')
plt.ylabel('Average House Price (£)')
plt.legend()
plt.title('Forecast of Average House Price in Bristol')
plt.xticks(rotation=45)  # Rotate the date labels for better readability
plt.tight_layout()
plt.show()
