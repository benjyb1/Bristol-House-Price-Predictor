import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
# import statsmodels.api as sm
import seaborn as sns
import os
from statsmodels.tsa.stattools import grangercausalitytests
import operator


def format_input_ward_data(file_path_1):
    """
    Format input ward data

    Inputs
    --------
    file_path_1 : File path for the ward dataset

    Returns
    --------
    ward_median_prices_pct_change : pandas dataframe of median ward prices in 3 month increments
                                    ROWS = median price as a % change from the previous row, each row represents a 3 month period
                                    COLS = ward names
    """
    # file_path_1 = r"C:\Users\acdal\OneDrive - University of Bristol\2024-2025\MDM3\Phase B\MDM3-Baggatron\Data\Raw\Bristol house prices by ward 1995-2023.xlsx"
    file_path = os.path.join(os.path.dirname(__file__), 'Data', 'Wards', 'Median_Prices.csv')
    ward_median_prices = pd.read_csv(file_path, header=0) # Extract the full dataset, ignoring the unnecessary labels before the actual table


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


def load_data():
    # Load ward data
    ward_median_prices_pct_change = format_input_ward_data(r"C:\Users\acdal\OneDrive - University of Bristol\2024-2025\MDM3\Phase B\MDM3-Baggatron\Data\Raw\Bristol house prices by ward 1995-2023.xlsx")
    
    # Extract ward names
    unique_wards = ward_median_prices_pct_change.columns.to_list()

    # Extract time labels
    time_points = ward_median_prices_pct_change.index
    
    return ward_median_prices_pct_change, unique_wards, time_points

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


def granger_causality(dataset,ward_i,ward_j, max_lag):
    """
    Perform the Granger causality test to determine if any values from ward_j are statistically significant in predicting future values for ward_i

    Inputs
    ----------
    dataset : pandas dataframe, contains the ward data that the Granger causality test is performed on
    ward_i : name of ward that we want to predict
    ward_j : name of ward that is being tested for any significance in predicting future ward_i values
    max_lag : maximum number of shifts tested for significance in predicting future ward_i values

    Returns
    ----------
    tuple(significant_values) : tuple of integers corresponding to the lags that can be used to predict future values for ward_i, 
                                i.e. which previous values from ward_j can be used to predict the next value for ward_i
    """
    
    granger_results = grangercausalitytests(dataset[[ward_i, ward_j]], maxlag = max_lag)
 
    significant_values = [] # list containing the lag values from ward_j that are useful for predicting future values of ward_i

    # iterate through all tested lags and see if any lags have statistical significance in predicting ward_i
    for lag in range(1,max_lag+1):
        ssr_ftest = granger_results[lag][0]['ssr_ftest'][1] # ssr based F test p value
        ssr_chi2test = granger_results[lag][0]['ssr_chi2test'][1] # ssr based chi2 test p value
        lrtest = granger_results[lag][0]['lrtest'][1] # likelihood ratio test p value
        params_ftest = granger_results[lag][0]['params_ftest'][1] # parameter F test p value
        p_values = [ssr_ftest, ssr_chi2test, lrtest, params_ftest]

        # current lag is deemed statistically significant if the p value for all four tests < 0.05 (5% significance level)
        if all(p < 0.001 for p in p_values):
            significant_values.append(lag)

    return tuple(significant_values)


def all_wards_granger_causality(dataset,unique_wards,max_lag):
    """
    Perform Granger causality test on all ward pairs in both directions

    Inputs
    ----------
    dataset : pandas dataframe containing all the ward data that the Granger causailty test is performed on
    unique_wards : list containing the names of all the wards in Bristol
    max_lag : maximum number of shifts tested for statisctical significance in predicting future ward_i values during the Granger causality test

    Returns
    ----------
    df_granger : pandas dataframe containing the results of the Granger causality test
                 row (index) = ward that is being predicted
                 column = ward that is being used to try and predict row ward
                 element = each cell contains a tuple of integer lags, which represent which previous values from the columns ward are statistically significant in predicting future values of the row ward 
    """
    df_granger = pd.DataFrame(index = unique_wards, columns = unique_wards)

    for i in unique_wards:
        for j in unique_wards:
            df_granger.at[i,j] = granger_causality(dataset, i, j, max_lag)
    return df_granger


def main():

    ward_median_prices_pct_change, unique_wards, time_points = load_data()
    
    # granger_causality_results = granger_causality(ward_median_prices_pct_change,'Ashley','Stoke Bishop',35)
    # granger_causality_results_reversed = granger_causality(ward_median_prices_pct_change,'Stoke Bishop','Ashley',35)
    
    # print(granger_causality_results)
    # print("RESULTS IN REVERSE: \n")
    # print(granger_causality_results_reversed)

    df_granger = all_wards_granger_causality(ward_median_prices_pct_change, unique_wards, max_lag = 35)
    name = 'ward_granger_causality_' + str()
    df_granger.to_csv('ward_granger_causality.csv')

    print('DONE!')


if __name__ == '__main__':
    main()