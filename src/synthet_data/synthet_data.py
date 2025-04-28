import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# Main function to generate accidents_varied_2023.csv
def generate_varied_data(input_file='data/raw/accidents_2023.csv', output_file='data/raw/accidents_2023_varied.csv'):
    # Load the input file
    try:
        df = pd.read_csv(input_file)
        print(f"File loaded: {input_file}")
        print(f"Number of rows: {len(df)}")
    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
        return
    
    total_size = len(df)
    
    # Select 50% of the real data
    df_half, _ = train_test_split(df, train_size=0.5, random_state=np.random.randint(0, 1000))
    print(f"Real data selected: {len(df_half)} rows")
    
    # Create metadata for the dataframe
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_half)
    print("Metadata created automatically from dataframe")
    
    # Generate synthetic data (50% of the ORIGINAL size, not half of the selected data)
    synthetic_size = total_size - len(df_half)  # Calculer combien de lignes synthétiques sont nécessaires
    print(f"Generating {synthetic_size} synthetic rows to match original size of {total_size}")
    
    model = GaussianCopulaSynthesizer(metadata)
    model.fit(df_half)
    df_synthetic = model.sample(num_rows=synthetic_size)
    print(f"Synthetic data generated: {len(df_synthetic)} rows")
    
    # Combine real and synthetic data
    df_combined = pd.concat([df_half, df_synthetic], ignore_index=True)
    print(f"Combined data: {len(df_combined)} rows (should be close to {total_size})")
    
    # Save the output file
    df_combined.to_csv(output_file, index=False)
    print(f"File saved: {output_file}")

if __name__ == "__main__":
    generate_varied_data()
