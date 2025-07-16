import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

np.random.seed(42)

# Load dataset
file_path = r"C:\Users\kitcr\Downloads\Bristol_house_prices_by_ward_ARIMA 1995-2023.xlsx"
df = pd.read_excel(file_path)

quarter_to_month = {'Q1': 3, 'Q2': 6, 'Q3': 9, 'Q4': 12}

# Convert Year_Quarter into a Date column
df['Year'] = df['Year_Quarter'].str[:4].astype(int)
df['Quarter'] = df['Year_Quarter'].str[-2:]
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' +
                            df['Quarter'].map(quarter_to_month).astype(str) + '-01')

df = df.iloc[:72] #toggle for 2008 data

# Define target and exogenous variables
target_series = df['Stoke Bishop']
#target_series = df['Lawrence Hill']
#target_series = df['Horfield']

#exogenous_features = df[['Windmill Hill', 'Redland']]
#exogenous_features = df[['Avonmouth and Lawrence Weston','Southmead', 'St George Central', 'Bedminster', 'St George West', 'Hengrove and Whitchurch Park', 'Hillfields', 'Frome Vale', 'Brislington West', 'Clifton Down']]
#exogenous_features = df[['Bishopsworth', 'St George Central', 'Bishopston and Ashley Down', 'Frome Vale', 'Windmill Hill']]

# 2008 optimal features
exogenous_features = df[['Windmill Hill', 'Redland', 'Horfield', 'Knowle']]
#exogenous_features = df[['Avonmouth and Lawrence Weston','Southmead']]
#exogenous_features = df[['Bishopsworth', 'St George Central', 'Bishopston and Ashley Down']]

# Scale the target variable
scaler = StandardScaler()
target_scaled = scaler.fit_transform(target_series.values.reshape(-1, 1)).flatten()

# Split into train and test
train_size = int(0.7 * len(target_scaled))
target_train, target_test = target_scaled[:train_size], target_scaled[train_size:]
exog_train, exog_test = exogenous_features.iloc[:train_size], exogenous_features.iloc[train_size:]

# Scale the exogenous features
scaler_exog = StandardScaler()
exog_train_scaled = scaler_exog.fit_transform(exog_train)
exog_test_scaled = scaler_exog.transform(exog_test)

exog_train_scaled = pd.DataFrame(exog_train_scaled, columns=exog_train.columns, index=exog_train.index)
exog_test_scaled = pd.DataFrame(exog_test_scaled, columns=exog_test.columns, index=exog_test.index)

# Define p, d, q ranges
p_range = range(1, 10)
q_range = range(1, 10)
d = 1

results_grid = pd.DataFrame(index=p_range, columns=q_range)

best_test_r2 = float("-inf")
best_test1_r2 = float("-inf")
best_pq = None
best_pq1 = None

# Grid search for best p, q
for p in p_range:
    for q in q_range:
        try:
            model = ARIMA(target_train, order=(p, d, q), exog=exog_train_scaled)
            result = model.fit()

            model1 = ARIMA(target_train, order=(p, d, q))
            result1 = model1.fit()

            forecast = result.forecast(steps=len(target_test), exog=exog_test_scaled)
            forecast1 = result1.forecast(steps=len(target_test))

            test_r2 = r2_score(target_test, forecast)
            test1_r2 = r2_score(target_test, forecast1)

            results_grid.loc[p, q] = test_r2

            if test_r2 > best_test_r2:
                best_test_r2 = test_r2
                best_pq = (p, q)

            if test1_r2 > best_test1_r2:
                best_test1_r2 = test1_r2
                best_pq1 = (p, q)
        except:
            continue

results_grid = results_grid.astype(float)

# Output the best parameters
print(f"ARIMAX Best (p, q): {best_pq} with R2: {best_test_r2:.4f}")
print(f"ARIMA Best (p, q): {best_pq1} with R2: {best_test1_r2:.4f}")

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

# Fit final models
p, q = best_pq
p1, q1 = best_pq1
arimax_model = ARIMA(target_train, order=(p, d, q), exog=exog_train_scaled)
arimax_result = arimax_model.fit()

arima_model = ARIMA(target_train, order=(p1, d, q1))
arima_result = arima_model.fit()

forecast = arimax_result.forecast(steps=len(target_test), exog=exog_test_scaled)
forecast1 = arima_result.forecast(steps=len(target_test))


forecast = scaler.inverse_transform(forecast.to_numpy().reshape(-1, 1)).flatten()
forecast1 = scaler.inverse_transform(forecast1.reshape(-1, 1)).flatten()
target_test_actual = scaler.inverse_transform(target_test.reshape(-1, 1)).flatten()

# Final comparison plot
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], target_series, label="Actual", color="blue")
plt.plot(df['Date'].iloc[train_size:], forecast1, label="ARIMA Forecast", color="green")
plt.plot(df['Date'].iloc[train_size:], forecast, label="ARIMAX Forecast", color="red")
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average House Price (£)", fontsize=14)
plt.legend()
plt.xticks(rotation=45)
plt.show()

fig, ax1 = plt.subplots(figsize=(14, 7))  # Create figure and first axis
ax1.plot(df['Date'][train_size:], target_test_actual, label="Actual Test Values", color="blue")
ax1.plot(df['Date'][train_size:], forecast, label="Predicted Test Values", color="red")
ax1.set_ylabel('Average House Price (£)', fontsize=14)
ax1.set_xlabel('Year', fontsize=14)
ax1.legend()
plt.show()
