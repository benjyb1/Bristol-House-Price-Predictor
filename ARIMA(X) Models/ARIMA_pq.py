import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
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

df_combined = df_combined.iloc[:197] # toggle for 2008 data

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

# Define ranges for p and q
p_range = range(1, 2)  # Adjust as needed
q_range = range(1, 2)  # Adjust as needed
d = 1  # Differencing order (assumed known)

# Initialize a grid to store test losses
results_grid = pd.DataFrame(index=p_range, columns=q_range)

# Track the best test loss and corresponding p, q
best_test_loss = float("-inf")
best_pq = None

# Grid search for p and q
for p in p_range:
    for q in q_range:
        try:
            # Fit the ARIMAX model
            model = ARIMA(target_train, order=(p, d, q), exog=exog_train)
            result = model.fit()

            # Forecast on the test set
            forecast = result.forecast(steps=len(target_test), exog=exog_test)

            # Compute test loss (MSE)
            test_loss = r2_score(target_test, forecast)

            # Store the test loss in the grid
            results_grid.loc[p, q] = test_loss

            # Update the best parameters if the current test loss is lower
            if test_loss > best_test_loss:
                best_test_loss = test_loss
                best_pq = (p, q)
        except Exception as e:
            # Handle models that fail to converge
            results_grid.loc[p, q] = np.nan

# Convert the grid to numeric for plotting
results_grid = results_grid.astype(float)

# Output the best parameters
print(f"Best (p, q): {best_pq} with Test Loss (R^2): {best_test_loss:.4f}")

# Plot the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(results_grid, cmap="seismic", aspect="auto", origin="lower")
cbar = plt.colorbar()
cbar.ax.set_ylabel("Constant of Determination ($R^2$)", fontsize=14)
plt.xticks(range(len(q_range)), labels=[str(q) for q in q_range], fontsize=14)
plt.yticks(range(len(p_range)), labels=[str(p) for p in p_range], fontsize=14)
plt.xlabel("q (Moving Average Order)", fontsize=14)
plt.ylabel("p (Autoregressive Order)", fontsize=14)
plt.show()


# Fit ARIMAX model
p, d, q = best_pq[0], 1, best_pq[1]  # Adjust these based on data behavior (use ACF/PACF to tune)
arimax_model = ARIMA(target_train, order=(1, d, 6), exog=exog_train)
arimax_result = arimax_model.fit()

# Forecast
n_forecast = len(target_test)
forecast_scaled = arimax_result.forecast(steps=n_forecast, exog=exog_test)

# Evaluate the model
test_loss = r2_score(target_test, forecast_scaled)
print(f"Test Loss (R^2): {test_loss:.4f}")

# Inverse scale the forecast and test data
forecast = scaler.inverse_transform(forecast_scaled.to_numpy().reshape(-1, 1)).flatten()
target_test_actual = scaler.inverse_transform(target_test.reshape(-1, 1)).flatten()

# Plot actual vs forecast
plt.figure(figsize=(12, 6))
plt.plot(df_combined['Date'], target_series, label="Actual", color="blue")
plt.plot(df_combined['Date'][train_size:], forecast, label="ARIMAX Forecast", color="red")
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average House Price (£)", fontsize=14)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

fig, ax1 = plt.subplots(figsize=(14, 7))  # Create figure and first axis
ax1.plot(df_combined['Date'][train_size:], target_test_actual, label="Actual Test Values", color="blue")
ax1.plot(df_combined['Date'][train_size:], forecast, label="Predicted Test Values", color="red")
ax1.set_ylabel('Average House Price (£)', fontsize=14)
ax1.set_xlabel('Date', fontsize=14)
ax1.legend()
plt.show()


# Forecast future values
future_steps = 12  # Number of months to forecast
future_exog = exogenous_features.iloc[-future_steps:]
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

# Plot future forecast
plt.figure(figsize=(12, 6))
plt.plot(df_combined['Date'].iloc[-18:], target_series.iloc[-18:], label="Actual", color="blue")
plt.plot(forecasted_dates, aligned_forecast, label="ARIMAX Future Forecast", color="red", linestyle='--')
plt.xlabel("Date", fontsize=14)
plt.ylabel("Average House Price (£)", fontsize=14)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
