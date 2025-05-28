import pandas as pd
import os
import yaml
import time
import sys
from sklearn.preprocessing import StandardScaler
import joblib
import json

def load_config(config_path="config.yaml"):
    # Load the configuration file
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def prepare_data(config_path="config.yaml"):
    # Load configuration parameters
    config = load_config(config_path)
    year = config["data_extraction"]["year"]

    # Check if the marker file accidents_{year}_synthet.done exists
    marker_file = f"data/raw/accidents_{year}_synthet.done"
    if not os.path.exists(marker_file):
        print(f"Error: The marker file {marker_file} does not exist. synthet_data.py must be executed first.")
        return

    # Define paths for raw and processed data
    synthet_path = os.path.join("data/raw", f"accidents_{year}_synthet.csv")
    raw_path = os.path.join("data/raw", f"accidents_{year}.csv")
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Use the synthetic file if it exists, otherwise the original file
    if os.path.exists(synthet_path):
        data = pd.read_csv(synthet_path, low_memory=False, sep=';')
    else:
        data = pd.read_csv(raw_path, low_memory=False, sep=';')

    # Remove 'adr' column if it exists as it's not relevant for analysis
    if 'adr' in data.columns:
        data = data.drop('adr', axis=1)

    # --- Ajout pour victim_age ---
    # Assumant que 'an' (année accident) et 'an_nais' (année naissance) sont présentes.
    if 'an' in data.columns and 'an_nais' in data.columns:
        data['an'] = pd.to_numeric(data['an'], errors='coerce')
        data['an_nais'] = pd.to_numeric(data['an_nais'], errors='coerce')
        data['victim_age'] = data['an'] - data['an_nais']
        data['victim_age'] = data['victim_age'].clip(0, 100) # Gestion simple des outliers pour l'âge (0-100 ans)
        print("Feature 'victim_age' created and outliers handled.")
    else:
        print("Warning: Columns 'an' or 'an_nais' not found. 'victim_age' not created.")

    # --- Ajout pour atm ---
    if 'atm' in data.columns:
        data['atm'] = pd.to_numeric(data['atm'], errors='coerce')
        dico_atm = {
            1: 0,  # Normal
            2: 1,  # Pluie légère
            3: 1,  # Pluie forte
            4: 1,  # Neige - grêle
            5: 1,  # Brouillard - fumée
            6: 1,  # Vent fort - tempête
            7: 1,  # Temps éblouissant
            8: 0,  # Temps couvert
            9: 0,  # Autre
            -1: 0  # Non renseigné (considéré comme Normal par défaut)
        }
        data['atm'] = data['atm'].replace(dico_atm)
        print("Feature 'atm' transformed to binary (0: Normal, 1: Risqué).")
    else:
        print("Warning: Column 'atm' not found. Transformation not applied.")

    # Handle missing values by filling with the mode of each column
    # Ceci imputera les NaN créés par to_numeric si 'an', 'an_nais', 'atm' n'étaient pas numériques
    # ou les NaN de victim_age si 'an' ou 'an_nais' étaient NaN.
    data.fillna(data.mode().iloc[0], inplace=True)

    # Specifically ensure that 'grav' has no NaN values (devrait être déjà géré par fillna ci-dessus)
    if 'grav' in data.columns and data['grav'].isna().any():
        data = data.dropna(subset=['grav'])
    
    # --- Modification de la transformation de 'grav' ---
    if 'grav' in data.columns:
        # Convertir en numérique, remplacer les erreurs de conversion par -1 (pour mapper à non-grave)
        data['grav'] = pd.to_numeric(data['grav'], errors='coerce').fillna(-1).astype(int)
        # Binarisation : 1 pour grave (valeurs originales 2, 3), 0 pour non-grave (valeurs originales 1, 4, -1)
        data['grav'] = data['grav'].apply(lambda x: 1 if x in [2, 3] else 0)
        print("Feature 'grav' transformed to binary (0: non-severe, 1: severe).")
    else:
        print("Warning: Column 'grav' not found. Transformation not applied.")

    # Select numerical columns for normalization
    # Exclure 'grav' (cible) et 'atm' (maintenant binaire) de la normalisation
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    # Assurer que 'victim_age' est dans numerical_columns si elle a été créée et est numérique
    if 'victim_age' in data.columns and pd.api.types.is_numeric_dtype(data['victim_age']):
        if 'victim_age' not in numerical_columns:
             # Normalement select_dtypes devrait l'inclure si elle est float64 ou int64.
             # Cette ligne est une sécurité au cas où elle serait d'un autre type numérique non inclus par défaut.
             pass # Elle devrait déjà y être.
    
    columns_to_exclude_from_scaling = ['grav']
    if 'atm' in data.columns: # Exclure atm si elle existe et a été transformée
        columns_to_exclude_from_scaling.append('atm')

    numerical_columns_to_scale = [col for col in numerical_columns if col not in columns_to_exclude_from_scaling]

    if numerical_columns_to_scale:
        scaler = StandardScaler()
        data[numerical_columns_to_scale] = scaler.fit_transform(data[numerical_columns_to_scale])

        scaler_path = os.path.join(processed_dir, f"scaler_{year}.joblib")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

        numerical_columns_scaled_path = os.path.join(processed_dir, f"numerical_columns_scaled_{year}.json")
        with open(numerical_columns_scaled_path, 'w') as f:
            json.dump(numerical_columns_to_scale, f)
        print(f"List of scaled numerical columns saved to {numerical_columns_scaled_path}")

    output_path = os.path.join(processed_dir, f"prepared_accidents_{year}.csv")
    data.to_csv(output_path, index=False)
    print(f"Prepared data saved to {output_path}")
    with open(os.path.join(processed_dir, "prepared_data.done"), "w") as f:
        f.write("done\n")

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
