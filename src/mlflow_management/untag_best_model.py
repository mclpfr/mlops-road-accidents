import mlflow
from mlflow.tracking import MlflowClient
import os
import yaml
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_mlflow_config():
    """Fetches MLflow config from config.yaml."""
    try:
        config_path = Path('/opt/project/config.yaml')
        if not config_path.exists():
            config_path = Path(__file__).resolve().parents[2] / 'config.yaml'
        
        with config_path.open() as f:
            config = yaml.safe_load(f).get('mlflow', {})
        
        uri = config.get('tracking_uri')
        model_name = config.get('model_name')
        username = config.get('username')
        password = config.get('password')
        
        if not uri:
            raise ValueError("mlflow.tracking_uri not found in config.yaml")
        if not model_name:
            raise ValueError("mlflow.model_name not found in config.yaml")
            
        return uri, model_name, username, password
    except Exception as e:
        logging.error(f"Could not load MLflow config from config.yaml: {e}")
        return None, None, None, None

def untag_best_model():
    """
    Finds the model version with the 'best_model' tag and removes it.
    """
    tracking_uri, model_name, username, password = get_mlflow_config()
    if not tracking_uri or not model_name:
        logging.error("Aborting due to missing MLflow configuration.")
        return

    # Set credentials as environment variables for MLflow client to use
    if username and password:
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password
        logging.info("MLflow credentials set from config.")
    else:
        logging.warning("MLflow credentials not found in config.yaml. Proceeding without authentication.")

    logging.info(f"Connecting to MLflow at {tracking_uri} to manage model '{model_name}'")
    client = MlflowClient(tracking_uri=tracking_uri)

    # Find model versions with the 'best_model' tag
    try:
        # Search for versions of the model
        versions = client.search_model_versions(f"name='{model_name}'")
        
        tagged_versions = [v for v in versions if 'best_model' in v.tags]

        if not tagged_versions:
            logging.info(f"No version of model '{model_name}' is currently tagged as 'best_model'. Nothing to do.")
            return

        for version in tagged_versions:
            logging.info(f"Found version {version.version} of model '{model_name}' with 'best_model' tag. Removing tag.")
            try:
                client.delete_model_version_tag(
                    name=model_name,
                    version=version.version,
                    key='best_model'
                )
                logging.info(f"Successfully removed 'best_model' tag from version {version.version} of model '{model_name}'.")
            except Exception as e:
                logging.error(f"Failed to remove tag from version {version.version}: {e}")

    except Exception as e:
        logging.error(f"An error occurred while searching for model versions: {e}")
        return

if __name__ == "__main__":
    untag_best_model()
