import pandas as pd
import os
import logging
import yaml
import time
from sklearn.preprocessing import StandardScaler
import joblib
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def prepare_data(config_path="config.yaml"):
    """
    Prepare the data by processing the synthetized accidents data file.
    """
    # Load configuration
    config = load_config(config_path)
    year = config["data_extraction"]["year"]
    
    # Define paths - WORKDIR is /app, data volume is mounted at /app/data
    app_data_dir = '/app/data'
    raw_dir = os.path.join(app_data_dir, 'raw')
    processed_dir = os.path.join(app_data_dir, 'processed')
    synthet_path = os.path.join(raw_dir, f'accidents_{year}_synthet.csv')
    
    # Create output directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    print(f"Raw data directory: {raw_dir}")
    print(f"Processed data directory: {processed_dir}")
    
    # Load accidents data
    print(f"Loading data from {synthet_path}...")
    data = pd.read_csv(synthet_path, low_memory=False, sep=';', encoding='latin1')
    print(f"Loaded data shape: {data.shape}")
    
    # Keep only the first occurrence for each accident (to avoid duplicates)
    data = data.drop_duplicates(subset='Num_Acc', keep='first')
    print(f"After dropping duplicates: {data.shape}")
    
    # Define the features we want to keep
    features = ["catu", "sexe", "trajet", "catr", "circ", "vosp", "prof", 
               "plan", "surf", "situ", "lum", "atm", "col"]
    
    # Check if all required columns exist
    missing_columns = [col for col in features + ['grav'] if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in the dataset: {missing_columns}")
    
    # Keep only the required columns
    data = data[features + ['grav']].copy()
    
    # Handle missing values by filling with the mode of each column
    data.fillna(data.mode().iloc[0], inplace=True)
    
    # Ensure that 'grav' has no NaN values
    data = data.dropna(subset=['grav'])

    # Convert gravity to binary classification (0: not severe, 1: severe)
    # Gravity categories:
    # 1 – Indemne (unharmed)
    # 2 – Tué (killed)
    # 3 – Blessé hospitalisé (hospitalized injured)
    # 4 – Blessé léger (slightly injured)
    # Group 1 and 4 as not severe (0), group 2 and 3 as severe (1)
    data['grav'] = data['grav'].apply(lambda x: 1 if x in [2, 3] else 0)
    
    # Print summary
    print("\n=== Data Summary ===")
    print(f"Final shape: {data.shape}")
    print("\nClass distribution (0: not severe, 1: severe):")
    print(data['grav'].value_counts())
    
    # Define paths for scaler
    model_dir = '/app/models'  # As per docker-compose volume mount for models
    os.makedirs(model_dir, exist_ok=True)
    scaler_path = os.path.join(model_dir, f'scaler_{year}.joblib')
    logger.info(f"Scaler will be saved to: {scaler_path}")

    # Separate features and target
    X = data[features]
    y = data['grav']

    # Initialize and fit scaler on features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert scaled features back to DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=X.index)
    
    # Combine scaled features and target
    processed_data_scaled = pd.concat([X_scaled_df, y], axis=1)
    
    # Save the scaler
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    # Save the processed and scaled data
    logger.info(f"Columns to save (after scaling): {processed_data_scaled.columns.tolist()}")
    logger.info(f"Data shape before save (after scaling): {processed_data_scaled.shape}")
    logger.info(f"Data head before save (after scaling):\n{processed_data_scaled.head()}")
    
    output_file = f"prepared_accidents_{year}.csv"
    output_path = os.path.join(processed_dir, output_file)
    processed_data_scaled.to_csv(output_path, index=False)
    logger.info(f"Processed and scaled data saved to {output_path}")

    # Verify the file was created and has the right columns
    if os.path.exists(output_path):
        df_check = pd.read_csv(output_path, nrows=1)
        print(f"\n=== Output File Verification ===")
        print(f"Output file created at: {output_path}")
        print(f"Columns in output file: {df_check.columns.tolist()}")
        print(f"First row values: {df_check.iloc[0].to_dict()}")
    else:
        print(f"ERROR: Output file was not created at {output_path}")
    
    # Create done file
    done_file = os.path.join(processed_dir, "prepared_data.done")
    with open(done_file, "w") as f:
        f.write("done\n")
    print(f"\nDone file created at: {done_file}")
    
    return data

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
