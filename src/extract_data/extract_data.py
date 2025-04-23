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

    # Merge the datasets on the 'Num_Acc' column
    merged_data = caract.merge(usagers, on="Num_Acc").merge(vehicules, on="Num_Acc").merge(lieux, on="Num_Acc")

    # Limit the dataset to half its original size to optimize performance
    original_size = len(merged_data)
    half_size = original_size // 2
    print(f"Limiting the dataset to {half_size} rows (original size: {original_size} rows)")
    merged_data = merged_data.head(half_size)
    
    # Save the merged data to the output directory
    output_path = os.path.join(output_dir, f'accidents_{year}.csv')
    merged_data.to_csv(output_path, index=False)

    print(f'Download and merge completed: All CSV files for {year} are stored in "{output_dir}".')
    with open(os.path.join(output_dir, "extract_data.done"), "w") as f:
        f.write("done\n")

# Execute the function
if __name__ == "__main__":
    download_accident_data()
