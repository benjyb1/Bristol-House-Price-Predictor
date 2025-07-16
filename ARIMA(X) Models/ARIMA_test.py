import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt

np.random.seed(42)

# Load dataset
file_path = r"C:\Users\kitcr\Downloads\Bristol extracted house price index 1995-2024.xlsx"
df = pd.read_excel(file_path)

pca_file_path = r"C:\Users\kitcr\Downloads\pca_transformed_data.xlsx"
df_pca = pd.read_excel(pca_file_path)

# Merge datasets
columns_to_add = ['Date', 'AveragePrice']
df_selected_columns = df[columns_to_add]
df_combined = pd.concat([df_selected_columns, df_pca], axis=1)

df_combined['Date'] = pd.to_datetime(df_combined['Date'], format='%d.%m.%Y %H:%M:%S')
df_combined.to_excel('combined_pca_dataset.xlsx', index=False)

# Define target and exogenous variables
target_series = df_combined['AveragePrice']
exogenous_features = df_combined[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]

# Scale the target variable
scaler = StandardScaler()
target_scaled = scaler.fit_transform(target_series.values.reshape(-1, 1)).flatten()

# Split into train and test
train_size = int(0.8 * len(target_scaled))
target_train, target_test = target_scaled[:train_size], target_scaled[train_size:]
exog_train, exog_test = exogenous_features.iloc[:train_size], exogenous_features.iloc[train_size:]

# Fit ARIMAX model
p, d, q = 31, 1, 33
arimax_model = ARIMA(target_train, order=(p, d, q), exog=exog_train)
arimax_result = arimax_model.fit()

# Forecast
n_forecast = len(target_test)
forecast_scaled = arimax_result.forecast(steps=n_forecast, exog=exog_test)

# Evaluate the model
test_loss = mean_squared_error(target_test, forecast_scaled)
print(f"Test Loss (MSE): {test_loss:.4f}")

# Inverse scale the forecast and test data
forecast = scaler.inverse_transform(forecast_scaled.to_numpy().reshape(-1, 1)).flatten()
target_test_actual = scaler.inverse_transform(target_test.reshape(-1, 1)).flatten()

# Plot actual vs forecast
plt.figure(figsize=(12, 6))
plt.plot(df_combined['Date'], target_series, label="Actual", color="blue")
plt.plot(df_combined['Date'][train_size:], forecast, label="Forecast", color="red")
plt.xlabel("Date")
plt.ylabel("Average House Price (£)")
plt.legend()
plt.title("ARIMAX: Actual vs Forecasted House Prices")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize model evaluation on the test set
plt.figure(figsize=(12, 6))
plt.plot(df_combined['Date'][train_size:], target_test_actual, label="Actual Test Values", color="blue")
plt.plot(df_combined['Date'][train_size:], forecast, label="Predicted Test Values", color="red")
plt.fill_between(
    df_combined['Date'][train_size:],
    target_test_actual,
    forecast,
    color="gray",
    alpha=0.3,
    label="Error"
)
plt.xlabel("Date")
plt.ylabel("Average House Price (£)")
plt.title("ARIMAX: Evaluation on Test Data")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Forecast future values
future_steps = 31  # Number of months to forecast
future_exog = exogenous_features.iloc[-future_steps:]
print(future_exog)# Adjust as needed
future_forecast_scaled = arimax_result.forecast(steps=future_steps, exog=future_exog)

# Undo scaling first
future_forecast = scaler.inverse_transform(
    future_forecast_scaled.to_numpy().reshape(-1, 1)
).flatten()

last_actual_value = target_series.iloc[-1]  # Last observed value
forecast_offset = last_actual_value - future_forecast[0]
aligned_forecast = future_forecast + forecast_offset
# Generate future dates
forecasted_dates = [df_combined['Date'].iloc[-1] + DateOffset(months=i) for i in range(0, future_steps)]

# Forecast future values
future_steps_long = 62  # Number of months to forecast
future_exog_long = exogenous_features.iloc[-future_steps_long:]
future_forecast_long_scaled = arimax_result.forecast(steps=future_steps_long, exog=future_exog_long)

# Undo scaling first
future_forecast_long = scaler.inverse_transform(
    future_forecast_long_scaled.to_numpy().reshape(-1, 1)
).flatten()

last_actual_value = target_series.iloc[-1]  # Last observed value
forecast_offset_long = last_actual_value - future_forecast_long[0]
aligned_forecast_long = future_forecast_long + forecast_offset_long
# Generate future dates
forecasted_dates_long = [df_combined['Date'].iloc[-1] + DateOffset(months=i) for i in range(0, future_steps_long)]

# Plot future forecast
past_months = 62
plt.figure(figsize=(12, 6))
plt.plot(df_combined['Date'].iloc[-past_months:], target_series.iloc[-past_months:], label="Actual", color="blue")
plt.plot(forecasted_dates, aligned_forecast, label="Future Forecast", color="green")
plt.plot(forecasted_dates_long, aligned_forecast_long, label="Longer Term Future Forecast", color="red")
plt.xlabel("Date")
plt.ylabel("Average House Price (£)")
plt.legend()
plt.title("ARIMAX: Future Forecast of House Prices")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

