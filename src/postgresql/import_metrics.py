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
        experiment_name = f"traffic-incidents-{year}"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if not experiment:
            logger.error(f"MLflow experiment '{experiment_name}' not found")
            return
            
        logger.info(f"Found experiment: {experiment.name} (ID: {experiment.experiment_id})")

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
            "year": year
        }
        
        # Create DataFrame and import into PostgreSQL
        if metrics_record:
            metrics_df = pd.DataFrame([metrics_record])
            metrics_df.to_sql('best_model_metrics', engine, if_exists='replace', index=False)
            logger.info(f"Import successful: Best model metrics imported (version {best_model_version.version})")
            logger.info(f"Metrics: {metrics_record}")
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