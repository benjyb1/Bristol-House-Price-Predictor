import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import os

# File path
file_path = "/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Desktop/University Engmaths/MDM3/Baggatron/Median Bristol house prices by ward 1995-2023 1a only.csv"

# Load data
df = pd.read_csv(file_path, skiprows=4)

time_columns = [col for col in df.columns if col.startswith('Year ending')]

for col in time_columns:
    df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')


def parse_quarter_end(col_name):
    if not col_name.startswith('Year ending'):
        return None
    date_str = col_name.replace('Year ending ', '')
    return pd.to_datetime(date_str, errors='coerce')


data_dict = {parse_quarter_end(col): df[col].values for col in time_columns}

df_time = pd.DataFrame.from_dict(data_dict, orient='columns')

df_time.columns = pd.to_datetime(df_time.columns, errors='coerce')

ward_column = "Ward name" 
if ward_column not in df.columns:
    raise ValueError(f"Column '{ward_column}' not found in the data!")

df_time.index = df[ward_column]

ward_name = "Central"  # Replace 
if ward_name not in df_time.index:
    raise ValueError(f"Ward '{ward_name}' not found in the data!")

prices = df_time.loc[ward_name].dropna()

# Prepare data for regression
X = np.arange(len(prices)).reshape(-1, 1)  # Numeric index as X (time series)
y = prices.values  # Median prices as y

# Define range of alpha values and polynomial orders
alpha_values = [0.01, 0.1, 1, 10, 100]  # List of alpha values to iterate through
order_values = range(1, 10)  # Polynomial orders from 1 to 9

results = []  # To store results for plotting and analysis

# Iterate through both polynomial orders and alpha values
for order in order_values:
    for alpha in alpha_values:
        # Generate polynomial features
        poly = PolynomialFeatures(degree=order)
        X_poly = poly.fit_transform(X)
        
        # Train a Ridge regression model (with regularization)
        model = Ridge(alpha=alpha)
        model.fit(X_poly, y)
        
        coefficients = model.coef_  # Coefficients of polynomial terms
        intercept = model.intercept_  # Intercept

        # Format the polynomial equation
        equation_terms = [f"{coeff:.4f}*x^{i}" for i, coeff in enumerate(coefficients)]
        polynomial_equation = f"{intercept:.4f} + " + " + ".join(equation_terms[1:])  # Exclude x^0 term as it's intercept
        
        # Predict and calculate R-squared
        y_pred = model.predict(X_poly)
        r_squared = model.score(X_poly, y)
        
        # Calculate the error (Mean Squared Error)
        mse = np.mean((y - y_pred)**2)  # Mean Squared Error
        
        # Add results to the list
        results.append({
            "Order": order, 
            "Alpha": alpha,
            "R-squared": r_squared, 
            "MSE": mse,  # Add MSE to the results
            "Equation": polynomial_equation
        })
        
        # Print the polynomial equation, R-squared, and MSE
        print(f"Order {order}, Alpha {alpha} Polynomial Equation:")
        print(polynomial_equation)
        print(f"R-squared: {r_squared:.4f}")
        print(f"MSE: {mse:.4f}")  # Print the MSE
        print("-" * 50)
   
        # Plot the fit for each order and alpha
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.7)
        plt.plot(X, y_pred, color='red', label=f'Poly Order {order}, Alpha {alpha}')
        plt.title(f'Polynomial Regression of Order {order} for {ward_name} (Alpha={alpha})')
        plt.xlabel('Time Index')
        plt.ylabel('Median Price')
        plt.legend()
        plt.show()

# Store the results in a DataFrame
results_df = pd.DataFrame(results)

# Plot results for all alphas and orders
plt.figure(figsize=(12, 6))

# Loop through each polynomial order to plot for each alpha
for alpha in alpha_values:
    alpha_results = results_df[results_df['Alpha'] == alpha]  # Filter results for this alpha value
    
    # Plot R-squared values across polynomial orders for this alpha
    plt.plot(alpha_results['Order'], alpha_results['R-squared'], marker='o', label=f'Alpha = {alpha}')
    
# Customize the plot
plt.title(f'R-squared vs. Polynomial Order for Different Alphas ({ward_name})')
plt.xlabel('Polynomial Order')
plt.ylabel('R-squared')
plt.legend(title="Alpha Values")
plt.grid(True)

# Save the R-squared vs Polynomial Order plot
plt.savefig("/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Desktop/University Engmaths/MDM3/Baggatron/R-squared_vs_Polynomial_Order.png")

# Show the plot
plt.show()

# Save the results to a CSV file (including MSE column)
results_df.to_csv("/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Desktop/University Engmaths/MDM3/Baggatron/regression_results.csv", index=False)

print("File saved successfully to /Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Desktop/University Engmaths/MDM3/Baggatron/")
