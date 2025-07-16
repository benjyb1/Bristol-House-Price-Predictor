import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import os


def ARIMAX_constructor(ward,
                       features,
                       plot = False):
    """
    

    Parameters
    ----------
    ward : str
        Ward to predict prices over
    features : list
        List of wards to use as exogeneous features.
    plot : bool
        Whether to plot associated graphs or not
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
    
    # Truncate data for crisis test
    df = df.iloc[:64, :]
    print(df)
    
    target_series = df[ward]
    exogenous_features = df[features]

    # Scale the target variable
    scaler = StandardScaler()
    target_scaled = scaler.fit_transform(target_series.values.reshape(-1, 1)).flatten()

    # Split into train and test
    train_size = int(0.7 * len(target_scaled))
    target_train, target_test = target_scaled[:train_size], target_scaled[train_size:]
    exog_train, exog_test = exogenous_features.iloc[:train_size], exogenous_features.iloc[train_size:]

    scaler_exog = StandardScaler()
    exog_train_scaled = scaler_exog.fit_transform(exog_train)
    exog_test_scaled = scaler_exog.transform(exog_test)
    
    exog_train_scaled = pd.DataFrame(exog_train_scaled, columns=exog_train.columns, index=exog_train.index)
    exog_test_scaled = pd.DataFrame(exog_test_scaled, columns=exog_test.columns, index=exog_test.index)
    
    # Define ranges for p and q
    p_range = range(1, 10)
    q_range = range(1, 10)
    d = 2  # Differencing order
    
    # Initialize a grid to store test losses
    results_grid = pd.DataFrame(index=p_range, columns=q_range)
    
    # Track the best test loss and corresponding p, q
    best_test_loss = float("inf")
    best_pq = None
    
    # Grid search for p and q
    best_pq = [3, 7]
    
    # Convert the grid to numeric for plotting
    results_grid = results_grid.astype(float)
    
    # Output the best parameters
    print(f"Best (p, q): {best_pq} with Test Loss (MSE): {best_test_loss:.4f}")
    
    
    # Fit ARIMAX model
    p, d, q = best_pq[0], 2, best_pq[1]  # Adjust these based on data behavior (use ACF/PACF to tune)
    arimax_model = ARIMA(target_train, order=(p, d, q), exog=exog_train_scaled)
    arimax_result = arimax_model.fit()
    
    # Forecast
    n_forecast = len(target_test)
    forecast_scaled = arimax_result.forecast(steps=n_forecast, exog=exog_test_scaled)
    
    # Evaluate the model
    test_loss = mean_squared_error(target_test, forecast_scaled)
    test_r2 = r2_score(target_test, forecast_scaled)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    
    # Inverse scale the forecast and test data
    forecast = scaler.inverse_transform(forecast_scaled.to_numpy().reshape(-1, 1)).flatten()
    target_test_actual = scaler.inverse_transform(target_test.reshape(-1, 1)).flatten()
    
    errors = target_test_actual - forecast
    percentage_error = (errors / target_test_actual) * 100
    print(percentage_error)

    # ---------------------------
    # Plotting section
    if plot:
    
        # Fit final models
        p, q = best_pq
        arimax_model = ARIMA(target_train, order=(p, d, q), exog=exog_train_scaled)
        arimax_result = arimax_model.fit()
        arima_model = ARIMA(target_train, order=(p, d, q))
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
        
        fig.savefig('Stoke_Bishop_Crisis_GRaph.pdf', format='pdf')

    # important stuff that gets returned for the iterative testing
    print(f'test R2 {test_loss}, p & q: {best_pq}')
    return test_loss, best_pq[0], best_pq[1]


def main():
    # settings for the model before I starting fucking around with code structure
    # change these when running from file
    ward = 'Stoke Bishop'

    pca_features = ['Brislington East', 'Lawrence Hill', 'Eastville', 'Windmill Hill',
                    'Cotham']

    ARIMAX_constructor(ward, pca_features, plot=True)


if __name__ == '__main__':
    main()
