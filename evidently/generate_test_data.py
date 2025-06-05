import pandas as pd
import numpy as np
import os
from pathlib import Path

def modify_data(df, drift_factor=0.1):
    """Modify a dataframe to simulate data drift.
    
    Args:
        df: Input DataFrame
        drift_factor: How much to modify the data (0.0 to 1.0)
    """
    df = df.copy()
    
    # Only modify numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        if df[col].nunique() > 1:  # Only modify columns with more than one unique value
            # Add some random noise based on the drift factor
            noise = np.random.normal(0, df[col].std() * drift_factor, size=len(df))
            df[col] = df[col] + noise
            
            # For integer columns, round the values
            if df[col].dtype == 'int64':
                df[col] = df[col].round().astype('int64')
    
    return df

def create_test_scenarios():
    # Load the current data
    current_file = "/home/ubuntu/mlops-road-accidents/evidently/current/current_data.csv"
    df = pd.read_csv(current_file)
    
    # Create test directory if it doesn't exist
    test_dir = "/home/ubuntu/mlops-road-accidents/evidently/test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create different test scenarios with increasing drift
    drift_levels = [0.1, 0.3, 0.5, 0.7, 1.0]  # Different drift levels to test
    
    for level in drift_levels:
        # Create modified data
        df_modified = modify_data(df, drift_factor=level)
        
        # Save to test directory
        output_file = os.path.join(test_dir, f"test_data_drift_{level:.1f}.csv")
        df_modified.to_csv(output_file, index=False)
        print(f"Created test file with drift level {level}: {output_file}")

if __name__ == "__main__":
    create_test_scenarios()
