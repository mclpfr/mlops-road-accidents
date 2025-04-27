#!/usr/bin/env python3

import os
import pandas as pd
import sqlalchemy
import logging
import yaml
import mlflow
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None

def get_postgres_connection(config):
    """Establish a connection to PostgreSQL database."""
    pg_config = config["postgresql"]
    connection_string = f"postgresql://postgres:postgres@localhost:5432/road_accidents"
    return sqlalchemy.create_engine(connection_string)

def import_model_metrics(engine, config):
    """Import model metrics from MLflow to PostgreSQL."""
    try:
        # MLflow configuration
        mlflow_config = config["mlflow"]
        tracking_uri = mlflow_config["tracking_uri"]
        if not tracking_uri:
            logger.error("MLflow tracking URI not found in configuration")
            return
            
        # Configure MLflow authentication
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_config["username"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_config["password"]
        
        logger.info(f"Connecting to MLflow at {tracking_uri}")
        logger.info(f"Using username: {mlflow_config['username']}")
        
        # Connect to MLflow
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        
        # List available experiments
        experiments = client.search_experiments()
        logger.info(f"Found {len(experiments)} experiments:")
        for exp in experiments:
            logger.info(f"  - {exp.name} (ID: {exp.experiment_id})")
        
        # Get experiments
        year = config["data_extraction"]["year"]
        experiment_name = f"road-accidents-{year}"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if not experiment:
            logger.error(f"MLflow experiment '{experiment_name}' not found")
            return
            
        logger.info(f"Found experiment: {experiment.name} (ID: {experiment.experiment_id})")
            
        # Get runs and their metrics
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            logger.warning(f"No runs found for experiment {experiment_name}")
            return
            
        logger.info(f"Found {len(runs)} runs for experiment {experiment_name}")
            
        # Prepare data for import
        metrics_data = []
        for _, run in runs.iterrows():
            run_id = run["run_id"]
            run_info = client.get_run(run_id)
            
            # Extract main metrics
            metrics = run_info.data.metrics
            logger.info(f"Run {run_id} metrics: {metrics}")
            
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
        
        # Create DataFrame and import into PostgreSQL
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_sql('model_metrics', engine, if_exists='replace', index=False)
            logger.info(f"Import successful: {len(metrics_df)} metric records")
            # Display the imported data
            for i, record in enumerate(metrics_data):
                logger.info(f"Record {i+1}: {record}")
        else:
            logger.warning("No metrics to import")
    
    except Exception as e:
        logger.error(f"Error importing model metrics: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main function."""
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration")
        return
    
    # Establish connection to PostgreSQL
    engine = get_postgres_connection(config)
    
    # Import model metrics
    import_model_metrics(engine, config)
    
    logger.info("Metrics import completed")

if __name__ == "__main__":
    main() 