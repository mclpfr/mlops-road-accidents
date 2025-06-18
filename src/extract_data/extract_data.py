import requests
from bs4 import BeautifulSoup
import os
import yaml
import pandas as pd

def load_config(config_path="config.yaml"):
    # Load the configuration file
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def download_accident_data(config_path="config.yaml"):
    # Load configuration parameters
    config = load_config(config_path)
    # Ensure year is a string to handle both quoted and unquoted values in YAML
    year = str(config["data_extraction"]["year"])
    
    # Direct URLs for 2023 files
    if year == "2023":
        csv_links = {
            f'usagers-{year}.csv': 'https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/20241023-153328/usagers-2023.csv',
            f'vehicules-{year}.csv': 'https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/20241023-153253/vehicules-2023.csv',
            f'lieux-{year}.csv': 'https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/20241023-153219/lieux-2023.csv',
            f'caract-{year}.csv': 'https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/20241028-103125/caract-2023.csv'
        }
    else:
        # Original implementation for other years
        url = config["data_extraction"]["url"]
        response = requests.get(url)
        response.raise_for_status()  # Ensure request was successful
        soup = BeautifulSoup(response.text, 'html.parser')
        resource_links = soup.find_all('a', href=True)
        files_to_download = [f'usagers-{year}.csv', f'vehicules-{year}.csv', f'lieux-{year}.csv', f'caract-{year}.csv']
        csv_links = {
            file_name: link['href']
            for file_name in files_to_download
            for link in resource_links
            if file_name in link['href']
        }

    # Check if any links were found
    if not csv_links:
        print(f"No files found for the year {year}! Please verify the page and file names.")
        return

    # Create a directory to store all files
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    # Download each CSV file directly to the output directory
    for file_name, csv_link in csv_links.items():
        file_path = os.path.join(output_dir, file_name)
        with requests.get(csv_link, stream=True) as r:
            r.raise_for_status()  # Ensure the request was successful
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):  # Download in chunks to handle large files
                    f.write(chunk)

    # Load the downloaded CSV files from the output directory
    usagers = pd.read_csv(os.path.join(output_dir, f'usagers-{year}.csv'), sep=";", low_memory=False)
    vehicules = pd.read_csv(os.path.join(output_dir, f'vehicules-{year}.csv'), sep=";", low_memory=False)
    lieux = pd.read_csv(os.path.join(output_dir, f'lieux-{year}.csv'), sep=";", low_memory=False)
    caract = pd.read_csv(os.path.join(output_dir, f'caract-{year}.csv'), sep=";", low_memory=False)

    # Keep only necessary columns from each dataset
    usagers_cols = ['Num_Acc', 'grav', 'catu', 'sexe', 'trajet']  # Keep important columns from usagers
    vehicules_cols = [col for col in vehicules.columns if col not in ['grav']]  # Remove grav if exists
    lieux_cols = [col for col in lieux.columns if col not in ['grav']]  # Remove grav if exists
    caract_cols = [col for col in caract.columns if col not in ['grav']]  # Remove grav if exists

    # Select only necessary columns
    usagers = usagers[usagers_cols]
    vehicules = vehicules[vehicules_cols]
    lieux = lieux[lieux_cols]
    caract = caract[caract_cols]

    # Merge the datasets on the 'Num_Acc' column using outer join to preserve all rows
    merged_data = caract.merge(usagers, on="Num_Acc", how='outer')
    merged_data = merged_data.merge(vehicules, on="Num_Acc", how='outer')
    merged_data = merged_data.merge(lieux, on="Num_Acc", how='outer')
    
    # Debug: Afficher les colonnes disponibles après fusion
    print("Colonnes après fusion:", merged_data.columns.tolist())

    # Convert 'grav' column to numeric, handling any potential string values
    merged_data['grav'] = pd.to_numeric(merged_data['grav'].astype(str).str.strip('"'), errors='coerce')

    # Limit dataset to 200000 lines 
    original_size = len(merged_data)
    limit = 200000
    if original_size > limit:
        print(f"Limiting the dataset to {limit} rows (original size: {original_size} rows)")
        merged_data = merged_data.head(limit)
    else:
        print(f"Processing complete dataset of {original_size} rows")
    
    # Save the merged data to the output directory
    output_path = os.path.join(output_dir, f'accidents_{year}.csv')
    merged_data.to_csv(output_path, sep=';', index=False)
    
    # Check that the file has been properly written
    if os.path.exists(output_path):
        print(f'Download and merge completed: All CSV files for {year} are stored in "{output_dir}".')
        # Force filesystem synchronization
        os.sync() if hasattr(os, 'sync') else None
        # Create the marker file only after confirming that the data file exists
        with open(os.path.join(output_dir, "extract_data.done"), "w") as f:
            f.write("done\n")
        print(f'Created marker file: {os.path.join(output_dir, "extract_data.done")}')
    else:
        print(f'Error: Failed to create output file {output_path}')
        exit(1)

# Execute the function
if __name__ == "__main__":
    download_accident_data()
