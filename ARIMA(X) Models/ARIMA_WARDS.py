import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt

np.random.seed(42)

# Load dataset
file_path = r"C:\Users\kitcr\Downloads\Bristol_house_prices_by_ward_ARIMA 1995-2023.xlsx"
df = pd.read_excel(file_path)

quarter_to_month = {'Q1': 3, 'Q2': 6, 'Q3': 9, 'Q4': 12}

# Split Year_Quarter into year and quarter
df['Year'] = df['Year_Quarter'].str[:4].astype(int)
df['Quarter'] = df['Year_Quarter'].str[-2:]

# Create a new Date column using year and corresponding month of the quarter
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' +
                            df['Quarter'].map(quarter_to_month).astype(str) + '-01')
print(df.head())


# Define target and exogenous variables
target_series = df['Clifton']
#exogenous_features = df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]

# Scale the target variable
scaler = StandardScaler()
target_scaled = scaler.fit_transform(target_series.values.reshape(-1, 1)).flatten()

# Split into train and test
train_size = int(0.8 * len(target_scaled))
target_train, target_test = target_scaled[:train_size], target_scaled[train_size:]
#exog_train, exog_test = exogenous_features.iloc[:train_size], exogenous_features.iloc[train_size:]

# Fit ARIMAX model
p, d, q = 3, 0, 2 # Adjust these based on data behavior (use ACF/PACF to tune)
arimax_model = ARIMA(target_train, order=(p, d, q))
arimax_result = arimax_model.fit()

# Forecast
n_forecast = len(target_test)
forecast_scaled = arimax_result.forecast(steps=n_forecast)

# Evaluate the model
test_loss = mean_squared_error(target_test, forecast_scaled)
print(f"Test Loss (MSE): {test_loss:.4f}")

# Inverse scale the forecast and test data
forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
target_test_actual = scaler.inverse_transform(target_test.reshape(-1, 1)).flatten()

# Plot actual vs forecast
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], target_series, label="Actual", color="blue")
plt.plot(df['Date'][train_size:], forecast, label="Forecast", color="red")
plt.xlabel("Date")
plt.ylabel("Average House Price - Clifton (£)")
plt.legend()
plt.title("ARIMAX: Actual vs Forecasted House Prices")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize model evaluation on the test set
plt.figure(figsize=(12, 6))
plt.plot(df['Date'][train_size:], target_test_actual, label="Actual Test Values", color="blue")
plt.plot(df['Date'][train_size:], forecast, label="Predicted Test Values", color="red")
plt.fill_between(
    df['Date'][train_size:],
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
future_steps = 4  # Number of months to forecast
arimax_model_2 = ARIMA(target_series, order=(p, d, q))
arimax_result_2 = arimax_model.fit()
#future_exog = exogenous_features.iloc[-future_steps:]
future_forecast_scaled = arimax_result_2.forecast(steps=future_steps)

# Undo scaling first
future_forecast = scaler.inverse_transform(
    future_forecast_scaled.reshape(-1, 1)
).flatten()

last_actual_value = target_series.iloc[-1]  # Last observed value
forecast_offset = last_actual_value - future_forecast[0]
aligned_forecast = future_forecast + forecast_offset

forecasted_dates = [df['Date'].iloc[-1] + DateOffset(months=3*i) for i in range(0, future_steps)]

# Plot future forecast
plt.figure(figsize=(12, 6))
plt.plot(df['Date'].iloc[-4:], target_series.iloc[-4:], label="Actual", color="blue")
plt.plot(forecasted_dates, aligned_forecast, label="Future Forecast", color="green")
plt.xlabel("Date")
plt.ylabel("Average House Price (£)")
plt.legend()
plt.title("ARIMAX: Future Forecast of House Prices Clifton")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()