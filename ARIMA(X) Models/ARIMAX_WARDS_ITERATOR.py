from ARIMAX_FINAL_WARDS import ARIMAX_constructor
from Influential_Wards import analyze_ward_correlations
from ARIMA_WARDS_pq import ARIMA_constructor
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



def optimal_ward_count(ward, pca_features, limit=5, crisis=False):
    """

    Parameters
    ----------
    ward : str
        Ward to predict for

    pca_features : list
        List of ordered top wards determined by PCA analysis
    limit : int
        Maximum amount of wards to test

    Returns
    -------
    None.

    """
    
    mse_scores = []
    p_scores = []
    q_scores = []
    for x in range(1, limit+1):
        features = pca_features[:x]
        mse_result, p_result, q_result = ARIMAX_constructor(ward, features, crisis)
        mse_scores.append(mse_result)
        p_scores.append(p_result)
        q_scores.append(q_result)
        print(f'{x} wards complete')

    return mse_scores, p_scores, q_scores


def ARIMA_Results_Generator(crisis=False):
    # functions exactly the same as ARIMAX_Results_Generator but uses an ARIMA model instead
    file_path = os.path.join(os.path.dirname(__file__), 'Data', 'Wards', 'Median_Prices.csv')
    prices = pd.read_csv(file_path, index_col=0).T

    wards = prices.columns.tolist()
    ward_count = len(wards)
    print(wards)

    mse_scores = np.zeros(ward_count)
    best_p = np.zeros(ward_count)
    best_q = np.zeros(ward_count)

    for ward, x in zip(wards, range(0, ward_count)):
        mse_scores[x], best_p[x], best_q[x] = ARIMA_constructor(ward, crisis)

    print(mse_scores)
    print(best_p)
    print(best_q)

    print(wards)
    mse_output = pd.DataFrame(mse_scores, index=wards, columns=[0])
    p_output = pd.DataFrame(best_p, index=wards, columns=[0])
    q_output = pd.DataFrame(best_q, index=wards, columns=[0])

    mse_output.to_csv(os.path.join(os.path.dirname(__file__), 'Data', 'ARIMA_mse.csv'))
    p_output.to_csv(os.path.join(os.path.dirname(__file__), 'Data', 'ARIMA_p.csv'))
    q_output.to_csv(os.path.join(os.path.dirname(__file__), 'Data', 'ARIMA_q.csv'))

def ARIMAX_Results_Generator(feature_limit = 15, crisis=False):
    """
    Performs the ARIMAX model on every ward with exogenous feature wards up to the desired limit
    Uses pearson correlation to determine the top wards to include as features
    Saves the r^2 scores and hyperparamaters into the /Data folder once complete
    Inputs
    -------
    feature_limit : Maximum number of exogenous feature wards to include

    Returns
    -------
    None.

    """
    file_path = os.path.join(os.path.dirname(__file__), 'Data', 'Wards', 'Median_Prices.csv')
    prices = pd.read_csv(file_path, index_col=0).T

    wards = prices.columns.tolist()
    ward_count = len(wards)

    mse_scores = np.zeros([ward_count, feature_limit])
    best_p = np.zeros([ward_count, feature_limit])
    best_q = np.zeros([ward_count, feature_limit])
    print(mse_scores)
    
    # for all wards calculate the r-squared for each added feature
    for ward, x in zip(wards, range(0, ward_count)):
        # get the ordered list of wards by correlation
        features = analyze_ward_correlations(ward)
        correlations = list(features.values())
        print(correlations)
        features = list(features.keys())

        # get r squared scores and add them to the array
        mse_scores[x], best_p[x], best_q[x] = optimal_ward_count(ward, features, limit = feature_limit)
        print(mse_scores)
        print(best_p)
        print(best_q)

    print(mse_scores)
    print(best_p)
    print(best_q)

    # construct dataframes with optimal hyperparamaters and best mse scores
    mse_output = pd.DataFrame(mse_scores, index=wards, columns=range(1, feature_limit+1))
    p_output = pd.DataFrame(best_p, index=wards, columns=range(1, feature_limit+1))
    q_output = pd.DataFrame(best_q, index=wards, columns=range(1, feature_limit+1))

    # save results, use different file name is crisis testing is being done
    if crisis:
        mse_output.to_csv(os.path.join(os.path.dirname(__file__), 'Data', 'ARIMAX_mse_crisis_benjy.csv'))
        p_output.to_csv(os.path.join(os.path.dirname(__file__), 'Data', 'ARIMAX_p_crisis_benjy.csv'))
        q_output.to_csv(os.path.join(os.path.dirname(__file__), 'Data', 'ARIMAX_q_crisis_benjy.csv'))
    else:
        mse_output.to_csv(os.path.join(os.path.dirname(__file__), 'Data', 'ARIMAX_mse_benjy.csv'))
        p_output.to_csv(os.path.join(os.path.dirname(__file__), 'Data', 'ARIMAX_p_benjy.csv'))
        q_output.to_csv(os.path.join(os.path.dirname(__file__), 'Data', 'ARIMAX_q_benjy.csv'))
       

def main():
    ARIMAX_Results_Generator(feature_limit=15, crisis=True)
    ARIMA_Results_Generator()

if __name__ == '__main__':
    main()
        