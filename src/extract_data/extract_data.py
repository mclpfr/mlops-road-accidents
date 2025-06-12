import os
import yaml
import requests
import pandas as pd

def load_config(config_path="/app/config.yaml"):
    # Load the configuration file
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def download_accident_data(config_path="/app/config.yaml"):
    # Load configuration parameters
    config = load_config(config_path)
    year = config["data_extraction"]["year"]
    
    # Dataset ID for the road accidents data
    dataset_id = "53698f4ca3a729239d2036df"  # ID for the road accidents dataset (Bases de données annuelles des accidents corporels de la circulation routière)
    
    # Get the dataset's resources
    api_url = f"https://www.data.gouv.fr/api/1/datasets/{dataset_id}/"
    response = requests.get(api_url)
    response.raise_for_status()
    
    dataset = response.json()
    resources = dataset.get('resources', [])
    
    # List of required CSV files for the specified year
    files_to_download = [f'usagers-{year}.csv', f'vehicules-{year}.csv', 
                        f'lieux-{year}.csv', f'caract-{year}.csv']
    
    # Dictionary to store valid download links
    csv_links = {}
    for resource in resources:
        if any(file_name in resource.get('title', '') for file_name in files_to_download):
            file_name = next((f for f in files_to_download if f in resource.get('title', '')), None)
            if file_name:
                csv_links[file_name] = resource['url']
    
    # Check if any links were found
    if not csv_links:
        print(f"No files found for the year {year}! Please verify the year and dataset.")
        print(f"Available resources: {[r.get('title') for r in resources]}")
        return

    # Create a directory to store all files
    output_dir = "/app/data/raw"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading files to {output_dir}")

    # Download each CSV file
    for file_name, csv_link in csv_links.items():
        file_path = os.path.join(output_dir, file_name)
        print(f"Downloading {file_name} from {csv_link}")
        
        with requests.get(csv_link, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {file_name}")

    # Load the downloaded CSV files
    try:
        usagers = pd.read_csv(os.path.join(output_dir, f'usagers-{year}.csv'), sep=";", low_memory=False)
        vehicules = pd.read_csv(os.path.join(output_dir, f'vehicules-{year}.csv'), sep=";", low_memory=False)
        lieux = pd.read_csv(os.path.join(output_dir, f'lieux-{year}.csv'), sep=";", low_memory=False)
        caract = pd.read_csv(os.path.join(output_dir, f'caract-{year}.csv'), sep=";", low_memory=False)
        
        print(f"Successfully loaded all CSV files for year {year}")
        
        # Keep only necessary columns from each dataset
        usagers_cols = ['Num_Acc', 'grav', 'catu', 'sexe', 'trajet']
        vehicules_cols = [col for col in vehicules.columns if col not in ['grav']]
        lieux_cols = [col for col in lieux.columns if col not in ['grav']]
        caract_cols = [col for col in caract.columns if col not in ['grav']]

        # Select only necessary columns
        usagers = usagers[usagers_cols]
        vehicules = vehicules[vehicules_cols]
        lieux = lieux[lieux_cols]
        caract = caract[caract_cols]

        # Merge the datasets
        merged_data = caract.merge(usagers, on="Num_Acc", how='outer')
        merged_data = merged_data.merge(vehicules, on="Num_Acc", how='outer')
        merged_data = merged_data.merge(lieux, on="Num_Acc", how='outer')
        
        # Limit dataset to 20000 rows
        print(f"Original dataset size: {len(merged_data)} rows")
        if len(merged_data) > 20000:
            print("Limiting dataset to 20000 rows")
            merged_data = merged_data.head(20000)
        
        # Save the merged data
        output_path = os.path.join(output_dir, f'accidents_{year}.csv')
        merged_data.to_csv(output_path, sep=';', index=False)
        print(f"Merged data saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing CSV files: {str(e)}")
        raise

if __name__ == "__main__":
    download_accident_data()
