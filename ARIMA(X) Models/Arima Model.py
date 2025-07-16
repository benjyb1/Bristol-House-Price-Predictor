import pandas as pd
from matplotlib import pyplot
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
df=pd.read_csv('/Users/benjyb/PycharmProjects/PythonProject1/.venv/bristol_average_house_prices.csv',header=0,parse_dates=[0],index_col=0)
df.index = pd.date_range(start=df.index[0], periods=len(df), freq='MS')
df_train = df.iloc[:-72] #splitting into Training and Testing data, last 6 years for testing
df_test = df.iloc[-72:]

def determine_params(df_train): #Determining P,D,Q for ARIMA model
    df_train_diff=df_train.diff().dropna()
    df_train_diff.plot()
    acf=plot_acf(df_train_diff)
    pacf=plot_pacf(df_train_diff) #Acf and Pacf shows its a 'Random Walk' time series
    plt.show()
    #Significance test to check its stationary in the first order
    adf=adfuller(df_train_diff)
    print(adf[1]) #1% so its stationary -- ARIMA(p,1,q)

    return
#From acf and pcaf, it looks like p=1,4 & q=4,7,10 even.
def residuals(model_fit):
    residuals=model_fit.resid[1:]
    fig,ax=plt.subplots(1,2)
    residuals.plot(ax=ax[0])
    ax[0].set_title('Residuals')
    residuals.plot(kind='kde',ax=ax[1])
    ax[1].set_title('Residuals KDE')
    plt.show()
    return

def show_best_model(p, q,title):
    model = ARIMA(df_train, order=(p, 1, q))
    model_fit = model.fit()
    forecast_test = model_fit.forecast(steps=72)

    # Add the forecast data as a new column
    df[f'forecast'] = [None] * len(df_train) + list(forecast_test)

    # Create a completely new plot for each iteration
    df.plot()  # Generates the plot for the dataframe
    plt.title(f"Best {title} with p={p}, q={q}")
    plt.show()
    return

def param_optimisation(p_range, q_range):
    best_rmse = float('inf')  # Initialize lowest RMSE
    best_aic = float('inf')  # Initialize lowest AIC
    best_bic = float('inf')  # Initialize lowest BIC

    best_rmse_params = None
    best_aic_params = None
    best_bic_params = None

    results = []  # To store all results for inspection later

    for p in range(0, p_range):
        for q in range(0, q_range):
            try:
                # Fit the ARIMA model
                model = ARIMA(df_train, order=(p, 1, q))
                model_fit = model.fit()

                # Forecast on the test set
                forecast_test = model_fit.forecast(steps=len(df_test))

                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(df_test, forecast_test))

                # Retrieve AIC and BIC
                aic = model_fit.aic
                bic = model_fit.bic

                # Store results for this combination
                results.append((p, q, rmse, aic, bic))

                # Update best parameters for RMSE
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_rmse_params = (p, q)

                # Update best parameters for AIC
                if aic < best_aic:
                    best_aic = aic
                    best_aic_params = (p, q)

                # Update best parameters for BIC
                if bic < best_bic:
                    best_bic = bic
                    best_bic_params = (p, q)

                #print(f"p={p}, q={q}, RMSE={rmse}, AIC={aic}, BIC={bic}")

            except Exception as e:
                # Gracefully handle errors
                print(f"Error for p={p}, q={q}: {e}")

    # Print out the best parameters for each criterion
    print("\nOptimization Results:")
    print(f"Best Parameters (RMSE): p={best_rmse_params[0]}, q={best_rmse_params[1]} with RMSE={best_rmse}")
    print(f"Best Parameters (AIC): p={best_aic_params[0]}, q={best_aic_params[1]} with AIC={best_aic}")
    print(f"Best Parameters (BIC): p={best_bic_params[0]}, q={best_bic_params[1]} with BIC={best_bic}")

    show_best_model(best_rmse_params[0],best_rmse_params[1],RMSE)
    show_best_model(best_aic_params[0], best_aic_params[1],AIC)
    show_best_model(best_bic_params[0], best_bic_params[1],BIC)
    # Return all results and best parameters for further use
    return results, best_rmse_params, best_aic_params, best_bic_params
#param_optimisation(5,10) #Comes out as 1,3, or 4,4

def model(p, q, forecast_years):
    """
    Fit an ARIMA model using the entire dataset and predict the next 'forecast_years'.

    Parameters:
    - p: ARIMA order parameter for AR terms
    - q: ARIMA order parameter for MA terms
    - title: Title for the plot
    - forecast_years: Number of years to forecast

    Returns:
    None
    """
    # Check the data frequency and calculate forecast steps
    data_frequency = 12  # Change to your dataset's actual frequency (e.g., 12 for monthly, 365 for daily)
    steps = forecast_years * data_frequency  # Total steps corresponding to forecast period

    # Fit the ARIMA model using the entire dataset
    model = ARIMA(df, order=(p, 1, q))
    model_fit = model.fit()

    # Make forecast for the next 'forecast_years'
    forecast = model_fit.forecast(st eps=steps)
    forecast_index = pd.date_range(start=df.index[-1], periods=steps + 1, freq=df.index.freq)[1:]
    forecast_df = pd.DataFrame({'forecast': forecast}, index=forecast_index)

    # Concatenate the forecast with the original data for plotting
    df_with_forecast = pd.concat([df, forecast_df])

    # Plot the original data and forecast
    df_with_forecast.plot()  # Plot all data (including forecast)
    plt.title(f"Forecast for Next {forecast_years} Years with p={p}, q={q}")
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.show()
    return
model(4,4,5)