import os
import logging
import yaml
import mlflow
import mlflow.sklearn
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    try:
        # Load the configuration file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using environment variables")
        # Fallback to environment variables
        config = {
            "data_extraction": {
                "year": os.getenv("DATA_YEAR", "2023"),
                "url": os.getenv("DATA_URL", "https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/")
            },
            "mlflow": {
                "tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
                "username": os.getenv("MLFLOW_TRACKING_USERNAME"),
                "password": os.getenv("MLFLOW_TRACKING_PASSWORD")
            },
        }
        return config

# Télécharger le meilleur modèle localement
def find_and_save_best_model(config_path="config.yaml"):
    try:
        # Load configuration parameters
        config = load_config(config_path)
        year = config["data_extraction"]["year"]
        logger.info(f"Loaded configuration for year {year}")

        # Configure MLflow from config.yaml or environment variables
        mlflow_config = config["mlflow"]
        tracking_uri = mlflow_config["tracking_uri"]
        if not tracking_uri:
            raise ValueError("MLflow tracking URI not found in config or environment variables")
            
        logger.info(f"Setting MLflow tracking URI to: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Configure authentication
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_config["username"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_config["password"]
        logger.info("MLflow authentication configured")

        # Test MLflow connection
        try:
            client = mlflow.tracking.MlflowClient()
            logger.info("Successfully connected to MLflow server")
            
            # Check existing models
            try:
                registered_models = client.search_registered_models()
                model_names = [model.name for model in registered_models]
                logger.info(f"Existing registered models: {model_names}")
                
                for model_name in model_names:
                    all_versions = client.search_model_versions(f"name='{model_name}'")
                    logger.info(f"Model {model_name} has {len(all_versions)} versions")
                    # Log only the latest version with best model tag
                    best_versions = [v for v in all_versions if v.tags and "best_model" in v.tags]
                    if best_versions:
                        best_version = best_versions[0]
                        logger.info(f"  Latest best version: {best_version.version}, Stage: {best_version.current_stage}")
                        logger.info(f"  Tags: {best_version.tags}")
                        
                        # Télécharger la version du modèle avec le tag "best_model"
                        model_uri = f"models:/{model_name}/{best_version.version}"
                        
                        # Charger le modèle
                        model = mlflow.sklearn.load_model(model_uri)
                        
                        # Créer le répertoire "models" s'il n'existe pas
                        os.makedirs("models", exist_ok=True)

                        # Sauvegarder le modèle localement au format .joblib
                        local_model_path = "models/best_model_2023.joblib"
                        joblib.dump(model, local_model_path)
                        logger.info(f"  Best Model downloaded and saved in {local_model_path}")
                        
            except Exception as e:
                logger.error(f"Error checking existing models: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to connect to MLflow server: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"An error occurred during best model finding: {str(e)}")

if __name__ == "__main__":
    find_and_save_best_model()
