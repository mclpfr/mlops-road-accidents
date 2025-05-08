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
        accidents_df = pd.read_csv(data_path, low_memory=False)
        
        columns_needed = [
            'Num_Acc', 'jour', 'mois', 'an', 'hrmn', 'lum', 'dep', 'com', 'agg', 'int', 'atm', 'col', 'adr', 'lat', 'long'
        ]
        
        available_columns = [col for col in columns_needed if col in accidents_df.columns]
        
        if not available_columns:
            logger.error("No required columns are available in the dataset")
            return
        
        accidents_df = accidents_df[available_columns]
        
        accidents_df.to_sql('accidents', engine, if_exists='replace', index=False, 
                           method='multi', chunksize=10000)
        
        logger.info(f"Import successful: {len(accidents_df)} accident records")
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
        
        year = config["data_extraction"]["year"]
        experiment_name = f"road-accidents-{year}"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if not experiment:
            logger.error(f"MLflow experiment '{experiment_name}' not found")
            return
            
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            logger.warning(f"No runs found for experiment {experiment_name}")
            return
            
        metrics_data = []
        for _, run in runs.iterrows():
            run_id = run["run_id"]
            run_info = client.get_run(run_id)
            
            metrics = run_info.data.metrics
            metrics_record = {
                "run_id": run_id,
                "run_date": datetime.fromtimestamp(run_info.info.start_time/1000.0),
                "model_name": "accident-severity-predictor",
                "accuracy": metrics.get("accuracy", 0),
                "precision_macro_avg": metrics.get("macro avg_precision", 0),
                "recall_macro_avg": metrics.get("macro avg_recall", 0),
                "f1_macro_avg": metrics.get("macro avg_f1-score", 0),
                "model_version": run_info.data.tags.get("mlflow.runName", "unknown"),
                "year": year
            }
            metrics_data.append(metrics_record)
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_sql('best_model_metrics', engine, if_exists='replace', index=False)
            logger.info(f"Import successful: {len(metrics_df)} metric records")
        else:
            logger.warning("No metrics to import")
    
    except Exception as e:
        logger.error(f"Error importing model metrics: {str(e)}")

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
