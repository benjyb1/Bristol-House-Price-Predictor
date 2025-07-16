import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import os

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

def analyze_data(
    df, 
    exclude_columns=None,
    imputation_method='knn', 
    n_neighbors=5, 
    variance_threshold=None,  
    n_components=None,
    save_path=None,
    visualise=True,
):
    """
    Process dataset through imputation, normalization, and PCA with flexible feature extraction.
    
    Parameters:
    df : pandas DataFrame
        Input dataset
    exclude_columns : list, optional
        List of column names to exclude from the analysis
    imputation_method : str
        Imputation method ('knn', 'mean', 'zero', 'median', 'mode')
    n_neighbors : int
        Number of neighbors for KNN imputation
    variance_threshold : float, optional
        Cumulative variance threshold for feature selection
    n_components : int, optional
        Explicitly set number of principal components to retain
    save_path : str
        Directory path to save processed datasets
    """
    # First, automatically filter for numerical columns
    numerical_data = df.select_dtypes(include=[np.number])
    categorical_data = df.select_dtypes(exclude=[np.number])
    
    # Then exclude specific columns if provided
    if exclude_columns is not None:
        columns_to_exclude = [col for col in exclude_columns if col in numerical_data.columns]
        if columns_to_exclude:
            print("\nExcluding the following columns:", columns_to_exclude)
            numerical_data = numerical_data.drop(columns=columns_to_exclude)
    
    print("\n1. Initial data shape:", numerical_data.shape)
    print("\nColumns being analyzed:", numerical_data.columns.tolist())
    print("\nMissing values before imputation:")
    missing_counts = numerical_data.isnull().sum()
    print(missing_counts[missing_counts > 0])
    
    # Check for columns with all NaN values
    all_nan_cols = numerical_data.columns[numerical_data.isnull().all()].tolist()
    if all_nan_cols:
        print("\nColumns with all NaN values (will be dropped):")
        print(all_nan_cols)
        numerical_data = numerical_data.drop(columns=all_nan_cols)
    
    # Imputation
    if imputation_method == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_array = imputer.fit_transform(numerical_data)
    elif imputation_method == 'mean':
        imputed_array = numerical_data.fillna(numerical_data.mean())
    elif imputation_method == 'zero':
        imputed_array = numerical_data.fillna(0)
    elif imputation_method == 'median':
        imputed_array = numerical_data.fillna(numerical_data.median())
    elif imputation_method == 'mode':
        imputed_array = numerical_data.fillna(numerical_data.mode().iloc[0])
    else:
        raise ValueError("Imputation method must be one of: 'knn', 'mean', 'zero', 'median', 'mode'")
    
    # Create DataFrame with imputed values
    imputed_data = pd.DataFrame(
        imputed_array if imputation_method == 'knn' else imputed_array.values,
        columns=numerical_data.columns,
        index=numerical_data.index
    )
    
    print("\n2. Shape after imputation:", imputed_data.shape)
    
    # Check for constant columns
    std_dev = imputed_data.std()
    constant_cols = std_dev[std_dev == 0].index.tolist()
    if constant_cols:
        print("\nConstant columns after imputation (will be dropped):")
        print(constant_cols)
        imputed_data = imputed_data.drop(columns=constant_cols)
    
    print("\n3. Final shape after removing constant columns:", imputed_data.shape)
    
    # Normalization
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(imputed_data),
        columns=imputed_data.columns,
        index=imputed_data.index
    )
    
    # PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # Calculate cumulative variance and select important features
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Component selection logic
    if n_components is not None:
        selected_components = n_components
    elif variance_threshold is not None:
        selected_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
    else:
        selected_components = np.argmax(cumulative_variance_ratio >= 0.8) + 1
    
    # Extract feature importance
    feature_importance = []
    for i in range(selected_components):
        pc_loadings = pd.Series(
            abs(pca.components_[i]),
            index=scaled_data.columns
        )
        top_features = pc_loadings.sort_values(ascending=False).head()
        
        for feature, loading in top_features.items():
            feature_importance.append({
                'Principal Component': f'PC{i+1}',
                'Feature': feature,
                'Loading': loading,
                'Variance Explained': explained_variance_ratio[i] * 100
            })
    
    # Create DataFrame with feature importance
    feature_importance_df = pd.DataFrame(feature_importance)
    
    # Create DataFrame with selected important features
    important_feature_names = feature_importance_df['Feature'].unique()
    important_features_data = imputed_data[important_feature_names]
    
    # Create PCA DataFrame with selected components
    pca_df = pd.DataFrame(
        pca_result[:, :selected_components],
        columns=[f'PC{i+1}' for i in range(selected_components)],
        index=scaled_data.index
    )
    
    # Save datasets if path is provided
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Save imputed dataset with important features
        imputed_filename = os.path.join(save_path, f'{file_name} imputed_important_features.csv')
        important_features_data.to_csv(imputed_filename)
        print(f"\n{file_name} Imputed dataset with important features saved to: {imputed_filename}")
        
        # Save PCA transformed dataset
        pca_filename = os.path.join(save_path, f'{file_name} pca_transformed_data.csv')
        pca_df.to_csv(pca_filename)
        print(f"{file_name} PCA transformed dataset saved to: {pca_filename}")
        
        # Save feature importance to CSV
        importance_filename = os.path.join(save_path, f'{file_name} feature_importance.csv')
        feature_importance_df.to_csv(importance_filename, index=False)
        print(f"{file_name} feature importance details saved to: {importance_filename}")
    
    # Print feature importance summary
    print("\n4. Feature Importance Summary:")
    for pc in feature_importance_df['Principal Component'].unique():
        pc_data = feature_importance_df[feature_importance_df['Principal Component'] == pc]
        print(f"\n{pc}:")
        print(f"  Variance Explained: {pc_data['Variance Explained'].iloc[0]:.2f}%")
        print("  Top Features:")
        for _, row in pc_data.iterrows():
            print(f"    - {row['Feature']}: {row['Loading']:.4f}")
    
    if visualise:
        try:
            visualize_pca_results(pca, scaled_data, feature_importance_df)
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    print("User chosen removed columns:\n", exclude_columns)
    print("Categorical data removed:\n", categorical_data.head(1))
    print("User chosen imputation method", imputation_method)

    return important_features_data, pca_df, pca, feature_importance_df


def visualize_pca_results(pca, scaled_data, feature_importance_df):
    """
    Create comprehensive visualizations for PCA analysis.
    """
    plt.figure(figsize=(15, 12))
    
    # 1. Scree Plot (Variance Explained)
    plt.subplot(2, 2, 1)
    explained_variance_ratio = pca.explained_variance_ratio_ * 100
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), 
             cumulative_variance_ratio, 'r-o', linewidth=2)
    plt.title('Scree Plot: Variance Explained')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained (%)')
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    
    # 2. Feature Importance Heatmap
    plt.subplot(2, 2, 2)
    importance_pivot = feature_importance_df.pivot(
        index='Feature', 
        columns='Principal Component', 
        values='Loading'
    )
    sns.heatmap(importance_pivot, cmap='coolwarm', center=0, annot=True, cbar=True)
    plt.title('Feature Importance Heatmap')
    plt.tight_layout()
    
    # 3. Correlation Matrix
    plt.subplot(2, 2, 3)
    correlation_matrix = scaled_data.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix')
    
    # 4. Cumulative Variance Plot
    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), 
             cumulative_variance_ratio, 'b-o')
    plt.title('Cumulative Variance Explained')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance (%)')
    plt.axhline(y=80, color='r', linestyle='--')
    plt.axhline(y=90, color='g', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Desktop/University Engmaths/MDM3/Baggatron/PCA_NaN_Feature_Outputs/PCA_Analysis.png')
    plt.show()

# Load dataset
file_path = '/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Desktop/University Engmaths/MDM3/Baggatron/Bristol extracted house price index 1995-2024.csv'
df = pd.read_csv(file_path)
file_name = os.path.basename(file_path)
print(file_name)

# Analyze dataset
# Use this to exclude particular columns from the analysis, such as time series
columns_to_exclude = ['Date']
imputed_data, pca_data, pca, feature_importance = analyze_data(
    df, 
    exclude_columns=columns_to_exclude,
    imputation_method='knn',  # or 'mean' or 'zero' or 'median' or 'mode' 
    n_neighbors=5,
    save_path='/Users/macbook/Library/Mobile Documents/com~apple~CloudDocs/Desktop/University Engmaths/MDM3/Baggatron/PCA_NaN_Feature_Outputs')