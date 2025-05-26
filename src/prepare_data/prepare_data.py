import pandas as pd
import os
import yaml
import time
import sys
from sklearn.preprocessing import StandardScaler

def load_config(config_path="config.yaml"):
    # Load the configuration file
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def prepare_data(config_path="config.yaml"):
    # Load configuration parameters
    config = load_config(config_path)
    year = config["data_extraction"]["year"]

    # Check if the marker file accidents_{year}_synthet.done exists
    marker_file = f"data/raw/accidents_{year}_synthet.done"
    if not os.path.exists(marker_file):
        print(f"Error: The marker file {marker_file} does not exist. synthet_data.py must be executed first.")
        return

    # Define paths for raw and processed data
    synthet_path = os.path.join("data/raw", f"accidents_{year}_synthet.csv")
    raw_path = os.path.join("data/raw", f"accidents_{year}.csv")
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Use the synthetic file if it exists, otherwise the original file
    if os.path.exists(synthet_path):
        data = pd.read_csv(synthet_path, low_memory=False, sep=';')
    else:
        data = pd.read_csv(raw_path, low_memory=False, sep=';')

    # Remove 'adr' column if it exists as it's not relevant for analysis
    if 'adr' in data.columns:
        data = data.drop('adr', axis=1)

    # Handle missing values by filling with the mode of each column
    data.fillna(data.mode().iloc[0], inplace=True)

    # Specifically ensure that 'grav' has no NaN values
    if 'grav' in data.columns and data['grav'].isna().any():
        # Remove rows with 'grav' = NaN 
        data = data.dropna(subset=['grav'])

    # Convert gravity to binary classification (0: not severe, 1: severe)
    # Gravity categories:
    # 1 – Indemne (unharmed)
    # 2 – Tué (killed)
    # 3 – Blessé hospitalisé (hospitalized injured)
    # 4 – Blessé léger (slightly injured)
    # Group 1 and 4 as not severe (0), group 2 and 3 as severe (1)
    data['grav'] = data['grav'].apply(lambda x: 1 if x in [2, 3] else 0)

    # Select numerical columns for normalization (excluding 'grav' and non-numerical columns)
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    numerical_columns = [col for col in numerical_columns if col != 'grav']  # Exclude target

    if numerical_columns:
        # Normalize numerical features using StandardScaler
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Save the prepared data to the processed directory
    output_path = os.path.join(processed_dir, f"prepared_accidents_{year}.csv")
    data.to_csv(output_path, index=False)
    print(f"Prepared data saved to {output_path}")
    with open(os.path.join(processed_dir, "prepared_data.done"), "w") as f:
        f.write("done\n")

if __name__ == "__main__":
    # Load configuration parameters
    config = load_config()
    year = config["data_extraction"]["year"]
    
    # Wait for the accidents_{year}_synthet.done file if it doesn't exist yet
    marker_file = f"data/raw/accidents_{year}_synthet.done"
    max_wait_time = 300  # Maximum wait time in seconds (5 minutes)
    wait_interval = 10   # Check every 10 seconds
    wait_time = 0
    
    while not os.path.exists(marker_file) and wait_time < max_wait_time:
        print(f"Waiting for {marker_file} to be created... ({wait_time}/{max_wait_time} seconds)")
        time.sleep(wait_interval)
        wait_time += wait_interval
    
    if not os.path.exists(marker_file):
        print(f"Error: The marker file {marker_file} was not created within the wait time.")
        sys.exit(1)
    
    # Prepare the data
    prepare_data()
