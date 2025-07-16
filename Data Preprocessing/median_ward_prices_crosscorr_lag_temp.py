import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
# import statsmodels.api as sm
import seaborn as sns
import os


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
    i : name of first ward
    j : name of second ward
    dataset : pandas dataframe that ward i and ward j columns are extracted from

    Returns
    --------
    rolling_corr_subset : pandas dataframe containing extracted ward pair data
    """
    rolling_corr_subset = dataset[[i,j]]
    return rolling_corr_subset


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
    crosscorr : float, cross-correlation coefficient
    """

    # ward_1 = dataset.iloc[:,i]
    # ward_2 = dataset.iloc[:,j]
    ward_1 = dataset[x[1]]
    ward_2 = dataset[y[1]]

    # return ward_2.shift(lag)
    return (lag,float(ward_1.corr(ward_2.shift(lag))))


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
    """
    
    lag_matrix = np.empty((len(unique_wards),len(unique_wards)),dtype=int) # matrix of lags that give the greatest corresponding corr coefs

    # Perform cross-correlation on all ward pairs
    for x in enumerate(unique_wards):
        for y in enumerate(unique_wards):
            ward_lag_corr_coefs = [crosscorr(dataset, x, y, lag = i) for i in range(max_lag * -1, max_lag + 1)]

            # Extract lag that gives the greatest cross-correlation
            best_corr = max(ward_lag_corr_coefs, key=lambda x: abs(x[1]))
            best_lag = best_corr[0]
            lag_matrix[x[0],y[0]] = best_lag
            # print(f"The greatest lag and correlation between {x} and {y} is {best_corr}")
    return lag_matrix


def lag_matrix_heatmap(lag_matrix,unique_wards):
    """
    Generate a heatmap from the lag matrix
    """
    # sns.heatmap(lag_matrix, square=True, annot=True, xticklabels=unique_wards, yticklabels=unique_wards, cbar_kws={'label': 'lag  (3 month periods)'})
    sns.heatmap(lag_matrix, square=True, annot=False, xticklabels=unique_wards, yticklabels=unique_wards, cmap="YlGnBu", cbar_kws={'label': 'Pearson Correlation Coefficient'})
    plt.show()


def rolling_corr(rolling_corr_subset, window):
    # Function that calculates the rolling correlation coefficient
    # Inputs: rolling_corr_subset = dataframe containing the data on two wards that rolling correlation is performed on
    rolling_corr_coef = rolling_corr_subset.iloc[:,0].rolling(window).corr(rolling_corr_subset.iloc[:,1])
    # mean_rolling_corr_coef = rolling_corr_coef.mean(skipna = True)
    # return mean_rolling_corr_coef
    return rolling_corr_coef


def generate_rolling_corr_dataset(house_data, window, unique_wards, time_points):
    """
    Generate an xarray Dataset containing the rolling window cross-correlation values for all ward pairs
    
    Inputs
    --------
    house_data : pandas dataframe, ward data that rolling cross-correlation is performed on
    window : int, the size of the rolling window
    unique_wards : list, names of all the wards in Bristol - used as coordinates for two of the xarray DataArray dimensions
    time_points : index labels of house_data, used as coordinates for the third dimension of the xarray DataArray

    Returns
    --------
    rolling_data : xarray DatasArray; dim1 = ward i, dim2 = ward j, dim 3 = rolling window cross-correlation coefficient at different time points
    """

    rolling_data = xr.DataArray(coords = [("ward i", unique_wards), ("ward j", unique_wards), ("time", time_points)])
    # rolling_data = xr.DataArray(coords=[unique_wards, unique_wards, time_points], dims=["ward i", "ward j", "time"])

    for i in unique_wards:
        for j in unique_wards:
            rolling_corr_subset = pair_extraction(i,j,house_data)
            rolling_data.loc[i,j,:] = rolling_corr(rolling_corr_subset, window)

    return rolling_data


def rolling_plot_time_mean(file_path_rolling):
    """
    Plot the mean magnitude correlation coefficient from all ward pairs at each point in time

    """


    # file_path_rolling = os.path.join(os.path.dirname(__file__), 'Data', 'Wards', 'rolling_correlation_4_formatted.csv')
    df_window = pd.read_csv(file_path_rolling) 
    df_window_data = df_window.iloc[:,2:]

    times = df_window_data.columns.values.tolist()

    for time in times:
        # pd.to_numeric(df_1_year_window_data)
        df_window_data[time].astype(float)

    # Return the absolute value of each correlation to avoid positive and negative correlations cancelling each other
    df_abs_window = df_window_data.abs()
    
    # At each point in time, calculate mean correlation coefficient from all ward pairs 
    df_abs_window_mean = df_abs_window.mean(axis = 0,numeric_only=True)

    # print(df_1_year_window_data[-1])
    # print(df_abs_window_mean)
    
    plt.plot(df_abs_window_mean)

    plt.show()
    

# def rolling_ward_pair_mean_corr_heatmap(file_path_rolling):
    
#     # file_path_rolling = os.path.join(os.path.dirname(__file__), 'Data', 'Wards', 'rolling_correlation_4_formatted.csv')
#     df_window = pd.read_csv(file_path_rolling) 
#     df_window_data = df_window.iloc[:,2:]

#     times = df_window_data.columns.values.tolist()

#     for time in times:
#         # pd.to_numeric(df_1_year_window_data)
#         df_window_data[time].astype(float)

#     df_window_ward_pair_mean = df_window_data.mean(axis = 1,numeric_only=True)
#     # print(df_1_year_window_data[-1])


#     plt.show()


def main():

    # Load ward data
    ward_median_prices_pct_change = format_input_ward_data(r"C:\Users\acdal\OneDrive - University of Bristol\2024-2025\MDM3\Phase B\MDM3-Baggatron\Data\Raw\Bristol house prices by ward 1995-2023.xlsx")
    
    df_ward_no_lag_corr = ward_median_prices_pct_change.corr(method='pearson')
    print(type(df_ward_no_lag_corr))

    # Extract ward names
    unique_wards = ward_median_prices_pct_change.columns.to_list()


    # Generate dictionary where: KEYS = Ward names, Values = Correlation against all other wards for key ward 
    # PERCENTAGE CHANGE NOT RAW VALUES!!!!!!
    ward_corr_dict = {}

    for ward in range(len(unique_wards)):
        ward_corr_dict[unique_wards[ward]] = df_ward_no_lag_corr.iloc[ward]

    print(ward_corr_dict)

   
    # Plot heatmap
    ward_corr_heatmap_no_lag = lag_matrix_heatmap(df_ward_no_lag_corr,unique_wards)

    
    
    # # Extract time labels
    # time_points = ward_median_prices_pct_change.index
    
    # # Determine how many shifts are considered during cross-correlation
    # # max_lag = int(ward_median_prices_pct_change.shape[0]/2) 
    # max_lag = 3

    # # Identify timeshifts between wards that give the greatest cross-corr coef
    # lag_matrix = generate_lag_matrix(ward_median_prices_pct_change, unique_wards, max_lag)

    # lag_matrix_heatmap(lag_matrix,unique_wards)
    
    # window_sizes = [4,8,12,20,40] # Corresponds to window sizes of 1 year, 2 years, 3 years, 5 years and 10 years

    # for window in window_sizes: 
    #     rolling_corr_data = generate_rolling_corr_dataset(ward_median_prices_pct_change, window, unique_wards, time_points)
    #     # print(rolling_corr_data.coords["time"])
    #     # print(rolling_corr_data)
    #     df = rolling_corr_data.to_dataframe("time").unstack()

    #     name = "rolling_correlation_" + str(window)
    #     df.to_csv(name + ".csv")





    # file_path_rolling_4 = os.path.join(os.path.dirname(__file__), 'Data', 'Wards', 'rolling_correlation_4_formatted.csv')
    # file_path_rolling_8 = os.path.join(os.path.dirname(__file__), 'Data', 'Wards', 'rolling_correlation_8_formatted.csv')
    # file_path_rolling_12 = os.path.join(os.path.dirname(__file__), 'Data', 'Wards', 'rolling_correlation_12_formatted.csv')
    # file_path_rolling_20 = os.path.join(os.path.dirname(__file__), 'Data', 'Wards', 'rolling_correlation_20_formatted.csv')
    # file_path_rolling_40 = os.path.join(os.path.dirname(__file__), 'Data', 'Wards', 'rolling_correlation_40_formatted.csv')


    # rolling_plot_time_mean(file_path_rolling_4)
    # rolling_plot_time_mean(file_path_rolling_8)
    # rolling_plot_time_mean(file_path_rolling_12)
    # rolling_plot_time_mean(file_path_rolling_20)
    # rolling_plot_time_mean(file_path_rolling_40)




    print("Done!")

        


if __name__ == '__main__':
    main()
    



###############################################################################################################################
# DRAFT CODE
###############################################################################################################################

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




    # file_path_2 = os.path.join(os.path.dirname(__file__), 'Data', 'Wards', 'rolling_correlation_4.csv')
    # df_1_year_window = pd.read_csv(file_path_2, header=1) # Extract the full dataset, ignoring the unnecessary labels before the actual table

    # file_path_3 = os.path.join(os.path.dirname(__file__), 'Data', 'Wards', 'rolling_correlation_4_formatted.csv')
    # df_1_year_window = pd.read_csv(file_path_3) 


    # df_1_year_window_data = df_1_year_window.iloc[:,5:]

    # times = df_1_year_window_data.columns.values.tolist()

    # for time in times:
    #     # pd.to_numeric(df_1_year_window_data)
    #     df_1_year_window_data[time].astype(float)

    # df_1_year_window_data_2 = df_1_year_window_data.mean(axis = 0,numeric_only=True)
    # # print(df_1_year_window_data[-1])

    # plt.plot(df_1_year_window_data_2)

    # plt.show()



    # ######################################
    # #Rolling Window Cross-correlation test
    # test_subset = pair_extraction(0,1,ward_median_prices_pct_change)
    # window_sizes = [4,8,12,20,40] # Corresponds to window sizes of 1 year, 2 years, 3 years, 5 years and 10 years

    # # Reverse the row order to go from most recent price to oldest price (from 2023 - 1995)
    # # This is so that the rolling window is primarily composed of the more recent data instead of the older data
    # # test_subset = test_subset.iloc[::-1]

    # df = pd.DataFrame()
    


    # for window in window_sizes:
    #     # mean_rolling_corr = rolling_corr(test_subset,window)
    #     # print(f"The mean correlation coefficient for a {window/4} year rolling window is: {mean_rolling_corr}")
    #     df[window] = rolling_corr(test_subset,window)




# # Loaded variable 'df' from URI: c:\Users\acdal\OneDrive - University of Bristol\2024-2025\MDM3\Phase B\MDM3-Baggatron\Data\Wards\rolling_correlation_12_formatted.csv
# import pandas as pd
# df = pd.read_csv(r'c:\Users\acdal\OneDrive - University of Bristol\2024-2025\MDM3\Phase B\MDM3-Baggatron\Data\Wards\rolling_correlation_12_formatted.csv',header=1)

# df.rename(columns = {'time':'ward i', 'Unnamed: 1':'ward j'}, inplace=True)
# df.drop([0], inplace=True)








# def pair_extraction(i, j, dataset):
#     """
#     Extract ward data for cross-correlation pair
    
#     Inputs
#     --------
#     i : int, column index of first ward
#     j : int, column index of second ward
#     dataset : pandas dataframe that column i and j are extracted from

#     Returns
#     --------
#     cross_corr_subset : pandas dataframe containing extracted ward pair data
#     """
#     cross_corr_subset = dataset.iloc[:, [i, j]]
#     return cross_corr_subset