import pandas as pd
import os
import yaml
from sklearn.preprocessing import StandardScaler

def load_config(config_path="config.yaml"):
    # Load the configuration file
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def prepare_data(config_path="config.yaml"):
    # Load configuration parameters
    config = load_config(config_path)
    year = config["data_extraction"]["year"]

    # Define paths for raw and processed data
    varied_path = os.path.join("data/raw", f"accidents_{year}_varied.csv")
    raw_path = os.path.join("data/raw", f"accidents_{year}.csv")
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Use the varied file if it exists, otherwise the original file
    if os.path.exists(varied_path):
        data = pd.read_csv(varied_path, low_memory=False, sep=';')
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
    data['grav'] = data['grav'].apply(lambda x: 1 if x in [3, 4] else 0)

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
    prepare_data()
