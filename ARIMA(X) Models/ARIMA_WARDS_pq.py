import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt


def ARIMA_constructor(ward,
                      plot=False):
    """
    
    Parameters
    ----------
    ward : str
        Ward to construct and test model for
    plot : bool, optional
        Whether to plot the results of the testing. The default is False.

    Returns
    -------
    test_r2 : int
        R squared value corresponding to optimal hyperparamaters
    best_p : int
        Best 'p' hyperparameter
    best_q : int
        Best 'q' hyperparameter

    """
    
    # Load dataset
    file_path = os.path.join(os.path.dirname(__file__), 'Data', 'Wards', 'Bristol_house_prices_by_ward_ARIMA 1995-2023.xlsx')
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
    target_series = df[ward]
    #exogenous_features = df[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]
    
    # Scale the target variable
    scaler = StandardScaler()
    target_scaled = scaler.fit_transform(target_series.values.reshape(-1, 1)).flatten()
    
    # Split into train and test
    train_size = int(0.8 * len(target_scaled))
    target_train, target_test = target_scaled[:train_size], target_scaled[train_size:]
    #exog_train, exog_test = exogenous_features.iloc[:train_size], exogenous_features.iloc[train_size:]
    
    # Define ranges for p and q
    p_range = range(1, 6)  # Adjust as needed
    q_range = range(1, 6)  # Adjust as needed
    d = 0  # Differencing order (assumed known)
    
    # Initialize a grid to store test losses
    results_grid = pd.DataFrame(index=p_range, columns=q_range)
    
    # Track the best test loss and corresponding p, q
    best_test_loss = float("inf")
    best_pq = None
    
    # Grid search for p and q
    for p in p_range:
        for q in q_range:
            try:
                # Fit the ARIMAX model
                model = ARIMA(target_train, order=(p, d, q))
                result = model.fit()
    
                # Forecast on the test set
                forecast = result.forecast(steps=len(target_test))
    
                # Compute test loss (MSE)
                test_loss = mean_squared_error(target_test, forecast)
    
                # Store the test loss in the grid
                results_grid.loc[p, q] = test_loss
    
                # Update the best parameters if the current test loss is lower
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_pq = (p, q)
            except Exception as e:
                # Handle models that fail to converge
                results_grid.loc[p, q] = np.nan
    
    # Convert the grid to numeric for plotting
    results_grid = results_grid.astype(float)
    
    # Output the best parameters
    print(f"Best (p, q): {best_pq} with Test Loss (MSE): {best_test_loss:.4f}")
    
    if plot:
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(results_grid, cmap="seismic", aspect="auto", origin="lower")
        plt.colorbar(label="Test Loss (MSE)")
        plt.xticks(range(len(q_range)), labels=[str(q) for q in q_range])
        plt.yticks(range(len(p_range)), labels=[str(p) for p in p_range])
        plt.xlabel("q (Moving Average Order)")
        plt.ylabel("p (Autoregressive Order)")
        plt.title("ARIMAX Model Performance (Test Loss)")
        plt.show()
    
    
    # Fit ARIMAX model
    p, d, q = best_pq[0], 0, best_pq[1]  # Adjust these based on data behavior (use ACF/PACF to tune)
    
    arimax_model = ARIMA(target_train, order=(p, d, q))
    arimax_result = arimax_model.fit()
    
    # Forecast
    n_forecast = len(target_test)
    forecast_scaled = arimax_result.forecast(steps=n_forecast)
    
    # Evaluate the model
    test_loss = mean_squared_error(target_test, forecast_scaled)
    test_r2 = r2_score(target_test, forecast_scaled)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    
    # Inverse scale the forecast and test data
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
    target_test_actual = scaler.inverse_transform(target_test.reshape(-1, 1)).flatten()
    
    if plot:
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
    
    if plot:
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

    return test_r2, best_pq[0], best_pq[1]


def main():
    ward = 'Clifton'
    plot = True
    ARIMA_constructor(ward, plot)


if __name__ == '__main__':
    main()