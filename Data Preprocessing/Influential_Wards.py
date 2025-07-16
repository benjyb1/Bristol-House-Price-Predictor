import pandas as pd
import numpy as np
from scipy import stats
import os

def analyze_ward_correlations(target_ward):
    """
    Analyze correlations between a target ward and all other wards' house prices.
    
    Parameters:
    target_ward (str): Name of the ward to analyze
    
    Returns:
    dict: Top 5 most correlated wards with their R-squared values
    """
    # Read the CSV file
    # Assuming first column is years and subsequent columns are ward names
    filepath = os.path.join(os.path.dirname(__file__), 'Data', 'Wards', 'Median_Prices.csv')
    df = pd.read_csv(filepath, index_col=0).T
    print(df)
    # Dictionary to store R-squared values
    r_squared_values = {}
    
    # Get the target ward's data
    target_data = df[target_ward]
    
    # Calculate R-squared for each ward
    for ward in df.columns:
        if ward != target_ward:  # Skip comparing ward with itself
            # Calculate correlation coefficient
            r_value, _ = stats.pearsonr(target_data, df[ward])
            # Calculate R-squared
            r_squared = r_value ** 2
            r_squared_values[ward] = r_squared
    
    # Sort wards by R-squared value and get top 5
    top_wards = dict(sorted(r_squared_values.items(), 
                            key=lambda x: x[1], 
                            reverse=True))
    
    return top_wards


def main():

    # Target ward to analyze (MODIFY THIS)
    ward_to_analyze = "Ashley"
    
    # Run the analysis
    top_wards = analyze_ward_correlations(ward_to_analyze)
    
    # Print results
    print(f"\nMost influential wards for predicting {ward_to_analyze}'s house prices:")
    print("-" * 60)
    for ward, r_squared in top_wards.items():
        print(f"Ward: {ward:<30} R-squared: {r_squared:.4f}")


if __name__ == '__main__':
    main()

