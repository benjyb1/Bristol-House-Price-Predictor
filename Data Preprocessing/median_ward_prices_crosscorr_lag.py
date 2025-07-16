import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import statsmodels.api as sm
import seaborn as sns
import os
import matplotlib.pyplot as plt


def format_input_ward_data(ward_median_prices = None):
    """
    Format input ward data

    Inputs
    --------
    ward_median_prices : pandas dataframe containing ward data to be formatted
                         allows ward data from other areas to be formatted if imported elsewhere
                         if None Bristol ward prices are assumed
    Returns
    --------
    ward_median_prices_pct_change : pandas dataframe of median ward prices in 3 month increments
                                    ROWS = median price as a % change from the previous row, each row represents a 3 month period
                                    COLS = ward names
    """
    # allow a dataframe to be passed to format data that isn't bristol only
    try:
        if ward_median_prices == None:
            file_path = os.path.join(os.path.dirname(__file__), 'Data', 'Wards', 'Median_Prices.csv')
            # file_path_1 = r"C:\Users\acdal\OneDrive - University of Bristol\2024-2025\MDM3\Phase B\MDM3-Baggatron\Data\Raw\Bristol house prices by ward 1995-2023.xlsx"
            ward_median_prices = pd.read_csv(file_path) # Extract the full dataset, ignoring the unnecessary labels before the actual table
    # except: pass is deliberate, dataframes cannot be truth tested
    except:
        pass

    # # Convert all the house prices into float
    # for time_step in range(1,ward_median_prices.shape[1]):
    #     ward_median_prices.iloc[:,time_step] = ward_median_prices.iloc[:,time_step].astype(float)

    # Transpose dataframe: ROWS = median price for 3 month period, COLS = ward names
    ward_median_prices = ward_median_prices.T
    header = ward_median_prices.iloc[0]
    ward_median_prices = ward_median_prices[1:]
    ward_median_prices.columns = header

    # dataframe of ward median house prices as a percentage change from the previous row
    ward_median_prices_pct_change = ward_median_prices.pct_change().dropna(axis = 0, how='all')
    return ward_median_prices_pct_change


def ward_names(dataset):
    """
    Extract unique ward names from dataset

    Inputs
    --------
    dataset : pandas dataframe containing ward data

    Returns
    --------
    unique_wards : list, names of all the different wards in Bristol
    """
    # Extract all the ward names
    unique_wards = dataset.columns.to_list()
    return unique_wards


def pair_extraction(i, j, dataset):
    """
    Extract ward data for cross-correlation pair
    
    Inputs
    --------
    i : int, column index of first ward
    j : int, column index of second ward
    dataset : pandas dataframe that column i and j are extracted from

    Returns
    --------
    cross_corr_subset : pandas dataframe containing extracted ward pair data
    """
    cross_corr_subset = dataset.iloc[:, [i, j]]
    return cross_corr_subset


def crosscorr(dataset, x, y, lag=0):
    """ 
    Lag-N cross correlation

    Inputs
    ----------
    dataset : pandas.DataFrame containing ward data
    x,y : ward names
    lag : int, default 0
    
    Returns
    ----------
    crosscorr : list : [(int) lag, (float) correlation at given lag]
    """

    # ward_1 = dataset.iloc[:,i]
    # ward_2 = dataset.iloc[:,j]
    ward_1 = dataset[x[1]]
    ward_2 = dataset[y[1]]

    # return ward_2.shift(lag)
    return (lag,float(ward_1.corr(ward_2.shift(lag), method='pearson')))


def generate_lag_matrix(dataset, unique_wards, max_lag):
    """
    Generate matrix containing the timeshift/lag between ward pairs that gives the greatest cross-correlation coefficient
    
    Inputs
    --------
    dataset : pandas dataframe, ward data that cross-correlation is performed on
    max_lag : int, maximum lag that is considered during cross-correlation

    Returns
    --------
    lag_matrix : ndarray, matrix where the i,jth element is the lag between ward i and ward j that gives the greatest cross-corr coef
    coef_matrix : ndarray, matrix where the i,jth element is the cross-corr coef of the wards with corresponding lag
    """
    
    lag_matrix = np.empty((len(unique_wards), len(unique_wards)), dtype=int) # matrix of lags that give the greatest corresponding corr coefs
    coef_matrix = np.empty((len(unique_wards), len(unique_wards)), dtype=float) # matrix of highest corr coefs at corresponding lag

    # Perform cross-correlation on all ward pairs
    for x in enumerate(unique_wards):
        for y in enumerate(unique_wards):
            print(y)
            ward_lag_corr_coefs = [crosscorr(dataset, x, y, lag = i) for i in range(max_lag * -1, max_lag + 1)]

            # Extract lag that gives the greatest cross-correlation
            best_corr = max(ward_lag_corr_coefs, key=lambda x: abs(x[1]))
            best_lag = best_corr[0]
            lag_matrix[x[0],y[0]] = best_lag
            coef_matrix[x[0], y[0]] = best_corr[1]
            # print(f"The greatest lag and correlation between {x} and {y} is {best_corr}")
    return lag_matrix, coef_matrix


def generate_lag_timeseries(dataset, ward_i, ward_j, max_lag):
    """
    Calculates the timeshift that for the given ward pair that gives the greatest cross-correlation coefficient,
    iterating through all possible max_lag to determine where the instability comes from
    Error testing function

    Parameters
    ----------
    dataset : pandas dataframe, ward data that cross-correlation is performed on
    wardI : int, index 
    wardJ : TYPE
        DESCRIPTION.
    max_lag : int, maximum lag in each direction to iterate though to

    Returns
    -------
    None.

    """

    lag_series = np.empty(max_lag-1, dtype=int) # series of lags that give greatest corresponding corr coefs
    coef_series = np.empty(max_lag-1, dtype=float) # series of highest corr coefs at corresponding lag
    coef_all = np.empty(max_lag-1, dtype=float) # series of corr coefs at the highest lag tested

    # iterates through every lag value up to the given maximum, finding the greatest corr coef & lag for each
    for lag in range(1, max_lag):
        ward_lag_corr_coefs = [crosscorr(dataset, [0, ward_i], [0, ward_j], lag=i) for i in range(0, lag)]

        # extract best lag & corresponding corr coef
        best_corr = max(ward_lag_corr_coefs, key=lambda x: abs(x[1]))
        lag_series[lag-1] = best_corr[0]
        coef_series[lag-1] = best_corr[1]
        coef_all[lag-1] = ward_lag_corr_coefs[-1][1]

    return lag_series, coef_series, coef_all

def lag_matrix_heatmap(lag_matrix,unique_wards):
    """
    Generate a heatmap from the lag matrix
    """
    sns.heatmap(lag_matrix, square=True, annot=True, xticklabels=unique_wards,
                yticklabels=unique_wards, cbar_kws={'label': 'lag  (3 month periods)'},
                cmap='vlag')
    plt.show()


def coef_matrix_heatmap(coef_matrix, unique_wards):
    """
    Generate a heatmap from the ceofficient matrixc
    """
    sns.heatmap(coef_matrix, square=True, xticklabels=unique_wards,
                yticklabels=unique_wards, cbar_kws={'label': 'cross-correlation coefficient'},
                cmap='vlag', vmin=-1, vmax=1)
    plt.show()


def coef_series_plot(lag_series, coef_series, coef_all, max_lag):
    fig, ax = plt.subplots(ncols=2)
    x = range(0, max_lag-1)
    ax[0].plot(x, lag_series, color='teal')
    ax[0].set_xlabel('max lag tested')
    ax[0].set_ylabel('optimum lag')

    ax[1].plot(x, coef_series, color='teal', label='optimum lag')
    ax[1].plot(x, coef_all, color='red', label='max lag')
    ax[1].legend()
    ax[1].set_xlabel('max lag tested')
    ax[1].set_ylabel('corr coef')
    fig.show()


def london_bristol_import():
    
    bristol_ward_prices = format_input_ward_data()

    london_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Data', 'Wards', 'London_Median_Prices.csv'), thousands=',')
    london_ward_prices = format_input_ward_data(ward_median_prices = london_data)

    bristol_ward_prices['London'] = london_ward_prices['London']

    lag_series, coef_series, coef_all = generate_lag_timeseries(bristol_ward_prices, 'London', 'Clifton', 80)
    coef_series_plot(lag_series, coef_series, coef_all, 80)

def main():

    # Load ward data
    ward_median_prices_pct_change = format_input_ward_data()
    # Extract ward names
    unique_wards = ward_median_prices_pct_change.columns.to_list()

    # Determine how many shifts are considered during cross-correlation
    # There are 110 time points, any value >~100 may return an error
    max_lag = 80

    # Identify timeshifts between wards that give the greatest cross-corr coef
    # lag_matrix, coef_matrix = generate_lag_matrix(ward_median_prices_pct_change, unique_wards, max_lag)

    # find best lag and corr coef for 2 wards over every max lag (testing function)
    # lag_series, coef_series, coef_all = generate_lag_timeseries(ward_median_prices_pct_change, 'Lawrence Hill', 'Stoke Bishop', max_lag)
    # print(f'lag at highest corr coef: \n{lag_series}')
    # print(f'corr coef at corresponding lag: \n{coef_series}')
    # coef_series_plot(lag_series, coef_series, coef_all, max_lag)

    # lag_matrix_heatmap(lag_matrix,unique_wards)
    # coef_matrix_heatmap(coef_matrix, unique_wards)
    

if __name__ == '__main__':
    london_bristol_import()
    # main()
    


# def rolling_corr(cross_corr_subset, window):
#     # Function that calculates the rolling correlation coefficient
#     # Inputs: cross_corr_subset = dataframe containing the data on two wards that rolling correlation is performed on
#     rolling_corr_coef = cross_corr_subset.iloc[:,0].rolling(window).corr(cross_corr_subset.iloc[:,1])
#     mean_rolling_corr_coef = rolling_corr_coef.mean(skipna = True)
#     return mean_rolling_corr_coef


# test_subset = pair_extraction(0,1,ward_median_prices)
# window_sizes = [4,8,12,20,40] # Corresponds to window sizes of 1 year, 2 years, 3 years, 5 years and 10 years

# # Reverse the row order to go from most recent price to oldest price (from 2023 - 1995)
# # This is so that the rolling window is primarily composed of the more recent data instead of the older data
# # test_subset = test_subset.iloc[::-1]

# for window in window_sizes:
#     mean_rolling_corr = rolling_corr(test_subset,window)
#     print(f"The mean correlation coefficient for a {window/4} year rolling window is: {mean_rolling_corr}")
