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
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="/app/config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        # Default values
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
        # Read the header to get columns
        with open(data_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(";")
        columns_needed = [
            'Num_Acc', 'jour', 'mois', 'an', 'hrmn', 'lum', 'dep', 'com', 'agg', 'int', 'atm', 'col', 'adr', 'lat', 'long'
        ]
        available_columns = [col for col in columns_needed if col in header]
        if not available_columns:
            logger.error("No required columns are available in the dataset")
            return

        # Count total number of lines for progress bar
        total_lines = sum(1 for _ in open(data_path, 'r', encoding='utf-8')) - 1  # -1 for header
        chunk_size = 10000
        n_chunks = (total_lines // chunk_size) + 1
        
        # Read and clean the CSV in chunks
        valid_rows = []
        total = 0
        invalid = 0
        
        logger.info("Starting data import with progress bar...")
        chunks = pd.read_csv(data_path, sep=';', chunksize=chunk_size, low_memory=False, dtype=str, on_bad_lines='skip')
        with tqdm(total=total_lines, desc="Importing data", unit="rows") as pbar:
            for chunk in chunks:
                # Only keep columns needed and drop rows with missing required columns
                chunk = chunk[available_columns]
                before = len(chunk)
                chunk = chunk.dropna(subset=available_columns)
                after = len(chunk)
                invalid += (before - after)
                valid_rows.append(chunk)
                total += before
                pbar.update(before)
                
        if valid_rows:
            logger.info("Concatenating valid rows...")
            accidents_df = pd.concat(valid_rows, ignore_index=True)
            logger.info("Writing to database...")
            accidents_df.to_sql('accidents', engine, if_exists='replace', index=False, method='multi', chunksize=10000)
            logger.info(f"Import successful: {len(accidents_df)} accident records (skipped {invalid} invalid rows)")
        else:
            logger.warning("No valid accident data to import.")
    except Exception as e:
        logger.error(f"Error importing accident data: {str(e)}")

def import_model_metrics(engine, config):
    """Import model metrics from MLflow to PostgreSQL."""
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
        
        # Get the model name
        model_name = "accident-severity-predictor"
        
        # Search for all versions of the model
        all_versions = client.search_model_versions(f"name='{model_name}'")
        
        # Find the version tagged as best_model
        best_model_version = None
        for version in all_versions:
            if version.tags and "best_model" in version.tags:
                best_model_version = version
                break
        
        if not best_model_version:
            logger.warning("No model version found with best_model tag")
            return
            
        logger.info(f"Found best model version: {best_model_version.version}")
        
        # Get the run info for the best model
        run_info = client.get_run(best_model_version.run_id)
        metrics = run_info.data.metrics
        
        # Create metrics record
        metrics_record = {
            "run_id": best_model_version.run_id,
            "run_date": datetime.fromtimestamp(run_info.info.start_time/1000.0),
            "model_name": model_name,
            "accuracy": metrics.get("accuracy", 0),
            "precision_macro_avg": metrics.get("macro avg_precision", 0),
            "recall_macro_avg": metrics.get("macro avg_recall", 0),
            "f1_macro_avg": metrics.get("macro avg_f1-score", 0),
            "model_version": best_model_version.version,
            "year": config["data_extraction"]["year"]
        }
        
        # Create DataFrame and import into PostgreSQL
        metrics_df = pd.DataFrame([metrics_record])
        metrics_df.to_sql('model_metrics', engine, if_exists='replace', index=False)
        logger.info(f"Import successful: Best model metrics imported (version {best_model_version.version})")
        logger.info(f"Metrics: {metrics_record}")
    
    except Exception as e:
        logger.error(f"Error importing model metrics: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def clean_csv_file(input_path, output_path, expected_columns=None, sep=","):
    """Clean a CSV file by removing rows with the wrong number of columns."""
    try:
        with open(input_path, "r", encoding="utf-8") as infile:
            lines = infile.readlines()
        if not lines:
            logger.error(f"Input file {input_path} is empty.")
            return False
        header = lines[0]
        if expected_columns is None:
            expected_columns = len(header.strip().split(sep))
        cleaned_lines = [header]
        for i, line in enumerate(lines[1:], start=2):
            if len(line.strip().split(sep)) == expected_columns:
                cleaned_lines.append(line)
            else:
                logger.warning(f"Skipping line {i} in {input_path}: wrong number of columns")
        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.writelines(cleaned_lines)
        logger.info(f"Cleaned file saved to {output_path} ({len(cleaned_lines)-1} valid rows)")
        return True
    except Exception as e:
        logger.error(f"Error cleaning CSV file: {str(e)}")
        return False

def main():
    """Main function."""
    time.sleep(5)
    
    # Load configuration
    config = load_config()
    
    # Establish connection to PostgreSQL
    engine = get_postgres_connection(config)
    
    # Import accident data
    year = config["data_extraction"]["year"]
    data_path = f"/app/data/raw/accidents_{year}.csv"
    import_accidents_data(engine, data_path)
    
    # Import model metrics
    import_model_metrics(engine, config)
    
    logger.info("Data import completed successfully")

if __name__ == "__main__":
    main() 
