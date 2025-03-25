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
    raw_path = os.path.join("data/raw", f"accidents_{year}.csv")
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Load the raw CSV file
    data = pd.read_csv(raw_path, low_memory=False)

    # Handle missing values by filling with the mode of each column
    data.fillna(data.mode().iloc[0], inplace=True)

    # Simplify the 'grav' column: assuming 0-4 scale, map to binary (0 = grave, 1 = not grave)
    # For example, let's consider 0-2 as "not grave" (1) and 3-4 as "grave" (0)
    data['grav'] = data['grav'].apply(lambda x: 0 if x >= 3 else 1)

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

if __name__ == "__main__":
    prepare_data()
