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
    year = config["data_extraction"]["year"]
    url = config["data_extraction"]["url"]

    # Perform an HTTP GET request to fetch the webpage content
    response = requests.get(url)
    response.raise_for_status()  # Ensure request was successful

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all links on the page
    resource_links = soup.find_all('a', href=True)

    # List of required CSV files for the specified year
    files_to_download = [f'usagers-{year}.csv', f'vehicules-{year}.csv', f'lieux-{year}.csv', f'caract-{year}.csv']

    # Dictionary to store valid download links
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
    usagers_cols = ['Num_Acc', 'grav']  # Only keep Num_Acc and grav from usagers
    vehicules_cols = [col for col in vehicules.columns if col not in ['grav']]  # Remove grav if exists
    lieux_cols = [col for col in lieux.columns if col not in ['grav']]  # Remove grav if exists
    caract_cols = [col for col in caract.columns if col not in ['grav']]  # Remove grav if exists

    # Select only necessary columns
    usagers = usagers[usagers_cols]
    vehicules = vehicules[vehicules_cols]
    lieux = lieux[lieux_cols]
    caract = caract[caract_cols]

    # Merge the datasets on the 'Num_Acc' column
    merged_data = caract.merge(usagers, on="Num_Acc").merge(vehicules, on="Num_Acc").merge(lieux, on="Num_Acc")

    # Convert 'grav' column to numeric, handling any potential string values
    merged_data['grav'] = pd.to_numeric(merged_data['grav'].astype(str).str.strip('"'), errors='coerce')

    # Limit dataset to 2000 lines 
    original_size = len(merged_data)
    print(f"Limiting the dataset to 20000 rows (original size: {original_size} rows)")
    merged_data = merged_data.head(20000)
    
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
