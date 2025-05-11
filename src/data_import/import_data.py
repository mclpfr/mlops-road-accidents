#!/usr/bin/env python3

import os
import pandas as pd
import sqlalchemy
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
                "host": os.getenv("POSTGRES_HOST", "postgres"),
                "port": os.getenv("POSTGRES_PORT", "5432"),
                "user": os.getenv("POSTGRES_USER", "postgres"),
                "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
                "database": os.getenv("POSTGRES_DB", "road_accidents")
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

        # Required columns for accident data
        columns_needed = [
            'Num_Acc', 'jour', 'mois', 'an', 'hrmn', 'lum', 'dep', 'com', 'agg', 'int', 'atm', 'col', 'adr', 'lat', 'long'
        ]

        # Filter only available columns
        available_columns = [col for col in columns_needed if col in accidents_df.columns]

        # Check if any required columns are available
        if not available_columns:
            logger.error("No required columns are available in the dataset")
            return

        # Select only available columns for import
        accidents_df = accidents_df[available_columns]

        # Import accident data into the 'accidents' table
        accidents_df.to_sql('accidents', engine, if_exists='replace', index=False,
                           method='multi', chunksize=10000)

        # Log the number of records successfully imported
        logger.info(f"Import successful: {len(accidents_df)} accident records")
    except Exception as e:
        # Log any error during accident data import
        logger.error(f"Error importing accident data: {str(e)}")

def import_model_metrics(engine, config):
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

        model_name = "accident-severity-predictor"
        metrics_data = []
        best_model_version = None
        
        try:
            # Search for all model versions
            model_versions = client.search_model_versions(f"name='{model_name}'")
            
            # Search for the version tagged "best_model"
            for version in model_versions:
                if version.tags and "best_model" in version.tags:
                    best_model_version = version
                    logger.info(f"Found best model version: {version.version} with accuracy {version.tags['best_model']}")
                    break
            
            if best_model_version is None:
                logger.warning(f"No version tagged as 'best_model' found for model {model_name}")
                return
            
            # Get metrics only for the best version
            run_id = best_model_version.run_id
            model_version = best_model_version.version
            status = best_model_version.current_stage  # e.g. 'Production', 'Staging', etc.
            
            try:
                run_info = client.get_run(run_id)
                metrics = run_info.data.metrics
                metrics_record = {
                    "run_id": run_id,
                    "run_date": datetime.fromtimestamp(run_info.info.start_time/1000.0),
                    "model_name": model_name,
                    "accuracy": metrics.get("accuracy", 0),
                    "precision_macro_avg": metrics.get("macro avg_precision", 0),
                    "recall_macro_avg": metrics.get("macro avg_recall", 0),
                    "f1_macro_avg": metrics.get("macro avg_f1-score", 0),
                    "model_version": model_version,
                    "model_stage": status,
                    "year": year
                }
                metrics_data.append(metrics_record)
                logger.info(f"Retrieved metrics for best model version {model_version}")
            except Exception as e:
                logger.error(f"Could not retrieve metrics for best model run_id {run_id}: {str(e)}")
                return
                
        except Exception as e:
            logger.error(f"Could not retrieve model versions from MLflow Model Registry: {str(e)}")
            return

        # Import metrics only if there is data
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_sql('best_model_metrics', engine, if_exists='replace', index=False)
            logger.info(f"Import successful: best model version {model_version} metrics imported")
        else:
            logger.warning("No metrics to import")

    except Exception as e:
        # Log any error during model metrics import
        logger.error(f"Error importing model metrics: {str(e)}")

def main():
    time.sleep(5)
    
    # Check if the marker file train_model.done exists
    marker_file = "/app/models/train_model.done"
    max_wait_time = 300  # Maximum wait time in seconds (5 minutes)
    wait_interval = 10   # Check every 10 seconds
    wait_time = 0
    
    while not os.path.exists(marker_file) and wait_time < max_wait_time:
        logger.info(f"Waiting for {marker_file} to be created... ({wait_time}/{max_wait_time} seconds)")
        time.sleep(wait_interval)
        wait_time += wait_interval
    
    if not os.path.exists(marker_file):
        logger.error(f"Error: The marker file {marker_file} was not created within the wait time.")
        return

    config = load_config()

    engine = get_postgres_connection(config)

    year = config["data_extraction"]["year"]
    data_path = f"/app/data/raw/accidents_{year}.csv"
    import_accidents_data(engine, data_path)

    import_model_metrics(engine, config)

    logger.info("Data import completed successfully")
    
    # Clean up all marker files
    marker_files = [
        "/app/data/raw/extract_data.done",
        "/app/data/raw/accidents_{year}_synthet.done".format(year=year),
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

