#!/usr/bin/env python3

import os
import pandas as pd
import sqlalchemy
from sqlalchemy import text
import json
import logging
import time
import mlflow
import yaml
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="/app/config.yaml"):
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {
            "data_extraction": {"year": "2023"},
            "mlflow": {
                "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", ""),
                "username": os.getenv("MLFLOW_TRACKING_USERNAME", ""),
                "password": os.getenv("MLFLOW_TRACKING_PASSWORD", "")
            },
            "postgresql": {
                "host": "localhost",
                "port": "5432",
                "user": "postgres",
                "password": "postgres",
                "database": "road_accidents"
            }
        }

def get_postgres_connection(config):
    pg_config = config["postgresql"]
    connection_string = f"postgresql://{pg_config['user']}:{pg_config['password']}@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"
    return sqlalchemy.create_engine(connection_string)

def import_accidents_data(engine, data_path):
    try:
        logger.info(f"Importing accident data from {data_path}")
        try:
            accidents_df = pd.read_csv(data_path, low_memory=False, on_bad_lines='skip', sep=';')
        except TypeError:
            accidents_df = pd.read_csv(data_path, low_memory=False, error_bad_lines=False, sep=';')

        # Some malformed lines in the CSV file were ignored during import.
        logger.warning("Some malformed lines in the CSV file were ignored during import.")

        # Define all columns we expect in the CSV
        all_columns = [
            'Num_Acc', 'jour', 'mois', 'an', 'hrmn', 'lum', 'dep', 'com', 'agg', 'int', 
            'atm', 'col', 'adr', 'lat', 'long', 'grav', 'catu', 'sexe', 'trajet', 'id_vehicule',
            'num_veh', 'senc', 'catv', 'obs', 'obsm', 'choc', 'manv', 'motor', 'occutc', 'catr',
            'voie', 'v1', 'v2', 'circ', 'nbv', 'vosp', 'prof', 'pr', 'pr1', 'plan', 'lartpc',
            'larrout', 'surf', 'infra', 'situ', 'vma'
        ]

        # Clean column names (remove any whitespace)
        accidents_df.columns = accidents_df.columns.str.strip()
        
        # Filter only columns that exist in the CSV
        available_columns = [col for col in all_columns if col in accidents_df.columns]
        
        # Log missing columns
        missing_columns = [col for col in all_columns if col not in accidents_df.columns]
        if missing_columns:
            logger.warning(f"The following columns are missing from the CSV: {', '.join(missing_columns)}")
            
        # Select and reorder columns
        accidents_df = accidents_df[available_columns]
        
        # Ensure data types are correct
        int_columns = ['jour', 'mois', 'an', 'lum', 'agg', 'int', 'atm', 'col', 'grav', 'catu', 
                      'sexe', 'trajet', 'senc', 'catv', 'obs', 'obsm', 'choc', 'manv', 'motor', 
                      'occutc', 'catr', 'circ', 'nbv', 'vosp', 'prof', 'plan', 'surf', 'infra', 'situ']
                      
        for col in int_columns:
            if col in accidents_df.columns:
                # Replace empty strings with NaN and then fill with a default value
                accidents_df[col] = pd.to_numeric(accidents_df[col].replace('', '0'), errors='coerce').fillna(0).astype(int)
        
        # Convert float columns
        float_columns = ['lat', 'long']
        for col in float_columns:
            if col in accidents_df.columns:
                # Replace comma with dot for decimal and handle empty strings
                accidents_df[col] = accidents_df[col].replace(',', '.', regex=True).replace('', '0')
                accidents_df[col] = pd.to_numeric(accidents_df[col], errors='coerce').fillna(0.0)
                
        # Clean text columns
        text_columns = ['hrmn', 'dep', 'com', 'adr', 'voie', 'v1', 'v2', 'pr', 'pr1', 'lartpc', 'larrout', 'vma']
        for col in text_columns:
            if col in accidents_df.columns:
                accidents_df[col] = accidents_df[col].astype(str).str.strip()
                # Replace empty strings with None for text fields
                accidents_df[col] = accidents_df[col].replace('', None)
                
        # Ensure Num_Acc and num_veh are not null
        if 'Num_Acc' in accidents_df.columns and 'num_veh' in accidents_df.columns:
            accidents_df = accidents_df.dropna(subset=['Num_Acc', 'num_veh'])

        # Import accident data into the 'accidents' table
        accidents_df.to_sql('accidents', engine, if_exists='replace', index=False,
                           method='multi', chunksize=10000)

        # Log the number of records successfully imported
        logger.info(f"Import successful: {len(accidents_df)} accident records")
    except Exception as e:
        # Log any error during accident data import
        logger.error(f"Error importing accident data: {str(e)}")

def import_model_metrics(engine, config):
    logger.info("Starting model metrics import...")
    try:
        mlflow_config = config["mlflow"]
        tracking_uri = mlflow_config["tracking_uri"]
        if not tracking_uri:
            logger.error("MLflow tracking URI not found in configuration")
            return

        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_config["username"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_config["password"]

        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

        year = config["data_extraction"]["year"]
        experiment_name = f"traffic-incidents-{year}"
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if not experiment:
            logger.error(f"MLflow experiment '{experiment_name}' not found")
            return

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if runs.empty:
            logger.warning(f"No runs found for experiment {experiment_name}")
            return

        # Direct search for the best model version in the MLflow registry
        model_name = mlflow_config.get("model_name", "accident-severity-predictor")
        metrics_data = []
        try:
            # 1) Prefer versions in the "Production" stage (or "Staging" if specified)
            preferred_stage = mlflow_config.get("preferred_stage", "Production")

            # 1) Retrieve versions using the standard get_latest_versions API (more robust)
            try:
                model_versions = client.get_latest_versions(model_name, stages=[preferred_stage])
            except Exception as e:
                logger.warning(f"get_latest_versions a échoué: {str(e)} — tentative avec search_model_versions")
                model_versions = []

            # 2) Fallback: use search_model_versions if no version exists in the desired stage
            if not model_versions:
                try:
                    model_versions = client.search_model_versions(f"name='{model_name}'")
                except Exception as e:
                    logger.error(f"Impossible d'interroger MLflow model versions: {str(e)}")
                    return

            if not model_versions:
                logger.error(
                    f"Aucune version du modèle '{model_name}' trouvée dans le registre MLflow")
                return

            # 3) Select the most recent version (highest version number)
            best_version = sorted(model_versions, key=lambda v: int(v.version), reverse=True)[0]

            run_id = best_version.run_id
            model_version = best_version.version
            model_stage = best_version.current_stage or 'None'

            # 4) Retrieve metrics from the associated run
            best_run = client.get_run(run_id)
            metrics = best_run.data.metrics

            logger.info(
                f"Sélection de la version {model_version} ({model_stage}) du modèle '{model_name}' — run {run_id}")
            logger.info(f"Clés des métriques disponibles: {list(metrics.keys())}")

            # 5) Build the metrics record
            metrics_record = dict(metrics)
            metrics_record.update({
                "run_id": run_id,
                "run_date": datetime.fromtimestamp(best_run.info.start_time / 1000.0),
                "model_name": model_name,
                "model_version": model_version,
                "model_stage": model_stage,
                "year": year,
            })
            metrics_data.append(metrics_record)

        except Exception as e:
            logger.error(f"Erreur lors du traitement des données MLflow: {str(e)}")
            return


        # Import metrics only if there is data
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            with engine.connect() as connection:
                connection.execute(text('DROP TABLE IF EXISTS best_model_metrics'))
            metrics_df.to_sql('best_model_metrics', engine, if_exists='replace', index=False)
            logger.info(f"Import successful: best model version metrics imported")
        else:
            logger.warning("No metrics to import")

    except Exception as e:
        # Log any error during model metrics import
        logger.error(f"Error importing model metrics: {str(e)}")


def main():
    time.sleep(5)
    
    config = load_config()
    engine = get_postgres_connection(config)

    year = config["data_extraction"]["year"]
    data_path = f"/app/data/raw/accidents_{year}.csv"
    
    import_accidents_data(engine, data_path)
    import_model_metrics(engine, config)


    # Clean up marker files after all imports are done
    marker_files = [
        "/app/data/raw/extract_data.done",
        f"/app/data/raw/accidents_{year}_synthet.done",
        "/app/data/processed/prepared_data.done",
        "/app/models/train_model.done"
    ]
    
    for marker_file in marker_files:
        if os.path.exists(marker_file):
            try:
                os.remove(marker_file)
                logger.info(f"Removed marker file: {marker_file}")
            except Exception as e:
                logger.warning(f"Failed to remove marker file {marker_file}: {str(e)}")
    
    logger.info("All marker files have been cleaned up")

if __name__ == "__main__":
    main()

