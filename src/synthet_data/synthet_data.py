import pandas as pd
import numpy as np

# Main function to generate accidents_2023_varied.csv
# This script creates a mixed dataset: 50% real data, 50% synthetic data (sampled from real distributions)
def generate_varied_data(input_file='data/raw/accidents_2023.csv', output_file='data/raw/accidents_2023_synthet.csv'):
    # Load the input file
    try:
        df = pd.read_csv(input_file, sep=';')
        print(f"File loaded: {input_file}")
        print(f"Number of rows: {len(df)}")
    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
        return
    
    total_size = len(df)
    # Select 50% of the real data (random sample)
    df_real = df.sample(frac=0.5, random_state=np.random.randint(0, 10000))
    n_synth = total_size - len(df_real)
    print(f"Real data selected: {len(df_real)} rows")
    print(f"Synthetic data to generate: {n_synth} rows")

    # Generate synthetic data by sampling each column independently from the real data (including Num_Acc)
    data_synth = {}
    for col in df.columns:
        # Sample values with replacement to keep the same distribution
        data_synth[col] = np.random.choice(df[col].dropna(), n_synth, replace=True)
    df_synth = pd.DataFrame(data_synth)
    print(f"Synthetic data generated: {len(df_synth)} rows")

    # Combine real and synthetic data
    df_combined = pd.concat([df_real, df_synth], ignore_index=True)
    # Shuffle the combined dataset
    df_combined = df_combined.sample(frac=1, random_state=np.random.randint(0, 10000)).reset_index(drop=True)
    print(f"Combined data: {len(df_combined)} rows (should be close to {total_size})")

    # Save the output file
    df_combined.to_csv(output_file, index=False, sep=';')
    print(f"File saved: {output_file}")

if __name__ == "__main__":
    generate_varied_data()
