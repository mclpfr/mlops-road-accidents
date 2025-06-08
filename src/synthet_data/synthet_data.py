import pandas as pd
import numpy as np
import os
import yaml
import time
import sys

def load_config(config_path="config.yaml"):
    # Load the configuration file
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Main function to generate accidents_{year}_varied.csv
# This script creates a mixed dataset: 50% real data, 50% synthetic data (sampled from real distributions)
def generate_varied_data(input_file=None, output_file=None, year=None):
    # Ensure different randomness for each execution of this script/function
    np.random.seed(time.time_ns() % (2**32 - 1)) # Seed with current nanoseconds

    # Check if input_file and output_file are provided
    if input_file is None or output_file is None or year is None:
        print("Error: input_file, output_file, and year must be provided.")
        return
    
    # Check if the marker file extract_data.done exists
    marker_file = f"data/raw/extract_data.done"
    if not os.path.exists(marker_file):
        print(f"Error: The marker file {marker_file} does not exist. extract_data.py must be executed first.")
        return
        
    # Load the input file
    try:
        df = pd.read_csv(input_file, sep=';', low_memory=False)
        print(f"File loaded: {input_file}")
        print(f"Number of rows: {len(df)}")
    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
        return
    
    total_size = len(df)
    # --- Generate base low-drift data (50% real, 50% resampled synthetic) ---
    print("--- Generating base low-drift data (50/50 method) ---")
    df_real = df.sample(frac=0.5, random_state=np.random.randint(0, 100000)) # Explicit random state for pandas
    n_synth = total_size - len(df_real)
    print(f"Real data selected: {len(df_real)} rows")
    print(f"Synthetic data to generate: {n_synth} rows")

    data_synth = {}
    for col in df.columns:
        non_na_values = df[col].dropna().values
        if len(non_na_values) > 0:
            data_synth[col] = np.random.choice(non_na_values, n_synth, replace=True)
        else:
            data_synth[col] = np.array([np.nan] * n_synth)
    df_synth = pd.DataFrame(data_synth)
    print(f"Base synthetic data generated: {len(df_synth)} rows")

    df_combined = pd.concat([df_real, df_synth], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=np.random.randint(0, 100000)).reset_index(drop=True) # Explicit random state for pandas shuffle
    print(f"Base combined data: {len(df_combined)} rows (should be close to {total_size})")

    # --- Specific modifications to induce drift in 3 target columns ---
    # Désactivé pour éviter la génération de dérive
    print("--- Drift-inducing modifications are disabled ---")
    print(f"--- Final data size: {len(df_combined)} rows ---")

    # Save the output file
    df_combined.to_csv(output_file, index=False, sep=';')
    print(f"File saved: {output_file}")
    
    # Create a marker file to indicate completion
    marker_file = os.path.join(os.path.dirname(output_file), f"accidents_{year}_synthet.done")
    with open(marker_file, "w") as f:
        f.write("done\n")
    print(f"Created marker file: {marker_file}")

if __name__ == "__main__":
    # Load configuration parameters
    config = load_config()
    year = config["data_extraction"]["year"]
    
    # Define input and output paths
    input_file = f"data/raw/accidents_{year}.csv"
    output_file = f"data/raw/accidents_{year}_synthet.csv"
    
    # Wait for the extract_data.done file if it doesn't exist yet
    marker_file = f"data/raw/extract_data.done"
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
    
    # Generate the synthetic data
    generate_varied_data(input_file=input_file, output_file=output_file, year=year)
