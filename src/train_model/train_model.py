import pandas as pd
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
import mlflow
import mlflow.sklearn
import json
import logging

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
            }
        }
        return config

def train_model(config_path="config.yaml"):
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
        except Exception as e:
            logger.error(f"Failed to connect to MLflow server: {str(e)}")
            raise

        # Create experiment if it doesn't exist
        experiment_name = f"road-accidents-{year}"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.info(f"Creating new experiment: {experiment_name}")
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            logger.info(f"Using experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Failed to set up experiment: {str(e)}")
            raise

        # Define paths for processed data and model storage
        data_path = f"data/processed/prepared_accidents_{year}.csv"
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)

        # Load the prepared data
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path, low_memory=False)

        # Select relevant features
        features = ["catu", "sexe", "trajet", "catr", "circ", "vosp", "prof", "plan", "surf", "situ", "lum", "atm", "col"]
        target = "grav"

        # Ensure all selected features exist in the dataset
        available_features = [col for col in features if col in data.columns]
        if not available_features:
            raise ValueError("None of the selected features are available in the dataset.")
        logger.info(f"Using features: {available_features}")

        # Prepare features (X) and target (y)
        X = pd.get_dummies(data[available_features], drop_first=True)
        y = data[target]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets")

        # Start an MLflow run
        with mlflow.start_run() as run:
            logger.info(f"Started MLflow run with ID: {run.info.run_id}")
            
            # Log parameters
            params = {
                "model_type": "RandomForestClassifier",
                "n_estimators": 100,
                "random_state": 42,
                "year": year,
                "features": json.dumps(available_features)
            }
            mlflow.log_params(params)
            logger.info("Logged model parameters")
            
            # Train a Random Forest Classifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            logger.info("Model training completed")

            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            logger.info("Model evaluation completed")
            
            # Log metrics
            mlflow.log_metric("accuracy", report["accuracy"])
            for label in report:
                if isinstance(report[label], dict):
                    for metric, value in report[label].items():
                        mlflow.log_metric(f"{label}_{metric}", value)
            logger.info("Logged model metrics")
            
            # Log model in MLflow
            mlflow.sklearn.log_model(model, "random_forest_model")
            logger.info("Model logged to MLflow")
            
            # Register model in Model Registry
            model_name = f"accident-severity-predictor"
            try:
                mlflow.register_model(
                    f"runs:/{run.info.run_id}/random_forest_model",
                    model_name
                )
                logger.info(f"Model registered in MLflow Model Registry with name: {model_name}")
            except Exception as e:
                logger.error(f"Failed to register model: {str(e)}")
                raise

            # Save the trained model locally as well
            model_path = f"{model_dir}/rf_model_{year}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Model saved locally to {model_path}")

    except Exception as e:
        logger.error(f"An error occurred during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()
