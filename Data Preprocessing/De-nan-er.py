import pandas as pd
import numpy as np
import math 

# Load the CSV file
file_path = '/Users/ted/Desktop/University Engmaths/MDM3/Baggatron/Ward borders .csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)


# Method 2: Replace with a specific value like 0
df = df.fillna(0)

# Save the updated DataFrame back to a CSV
output_file_path = '/Users/ted/Desktop/University Engmaths/MDM3/Baggatron/Bristol_ward_borders.csv'  # Path for saving the cleaned CSV
df.to_csv(output_file_path, index=False)

print(f"Processed CSV saved to {output_file_path}")
