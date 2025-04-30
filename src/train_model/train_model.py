import pandas as pd
import joblib
import yaml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
import mlflow
import mlflow.sklearn
import json
import logging
import random
import time
import traceback
import subprocess

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
            "model": {
                "type": "RandomForestClassifier",
                "hyperparameters": {
                    "n_estimators": [100],
                    "max_depth": [None],
                    "min_samples_split": [2],
                    "min_samples_leaf": [1],
                    "max_features": ["sqrt"],
                    "class_weight": [None]
                },
                "random_state": 42,
                "test_size": 0.2,
                "cv_folds": 5
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
            except Exception as e:
                logger.error(f"Error checking existing models: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to connect to MLflow server: {str(e)}")
            raise

        # Create experiment with a fixed name
        experiment_name = "traffic-incidents-2023"
        try:
            # Check if the experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is not None:
                # If the experiment is marked as deleted, permanently delete it
                if experiment.lifecycle_stage == "deleted":
                    logger.info(f"Experiment {experiment_name} exists but is deleted, permanently deleting it")
                    client.delete_experiment(experiment.experiment_id)
                else:
                    # Use the existing experiment
                    logger.info(f"Using existing experiment: {experiment_name}")
                    mlflow.set_experiment(experiment_name)
                    logger.info(f"Set experiment to: {experiment_name}")
            else:
                # Create a new experiment
                logger.info(f"Creating new experiment: {experiment_name}")
                mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
                logger.info(f"Created and set experiment to: {experiment_name}")
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
        
        if y.isna().any():
            logger.info(f"Found {y.isna().sum()} NaN values in target variable. Dropping these rows.")
            mask = ~y.isna()
            X = X[mask]
            y = y[mask]
            logger.info(f"After removing NaN values: {len(X)} samples remaining")

        # Use a fixed value for reproducibility
        logger.info(f"Random seed used for this run: 42")

        # Split the data into training and testing sets
        model_config = config.get("model", {})
        test_size = model_config.get("test_size", 0.2)
        random_state = model_config.get("random_state", 42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logger.info(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets")

        # Start an MLflow run without using the context manager to be able to end it explicitly
        active_run = mlflow.start_run()
        run = active_run
        logger.info(f"Started MLflow run with ID: {run.info.run_id} and name: {run.info.run_name}")
        try:
            logger.info(f"Started MLflow run with ID: {run.info.run_id}")
            
            # Entraînement direct sans grid search ni hyperparamètres multiples
            rf_params = {}
            if "hyperparameters" in model_config:
                # On prend le premier élément de chaque liste si présent
                for k, v in model_config["hyperparameters"].items():
                    if isinstance(v, list):
                        rf_params[k] = v[0]
                    else:
                        rf_params[k] = v
            rf_params["random_state"] = random_state
            model = RandomForestClassifier(**rf_params)
            logger.info(f"Training RandomForestClassifier with params: {rf_params}")
            model.fit(X_train, y_train)
            best_params = rf_params
            
            # Logging parameters
            params = {
                "model_type": model_config.get("type", "RandomForestClassifier"),
                "year": year,
                "features": json.dumps(available_features),
                "cv_folds": model_config.get("cv_folds", 5),
                "test_size": test_size,
                **best_params
            }
            mlflow.log_params(params)
            logger.info(f"Parameters used: {best_params}")
            
            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            logger.info("Model evaluation completed")
            
            # Log metrics
            current_accuracy = report["accuracy"]
            mlflow.log_metric("accuracy", current_accuracy)
            for label in report:
                if isinstance(report[label], dict):
                    for metric, value in report[label].items():
                        mlflow.log_metric(f"{label}_{metric}", value)
            logger.info("Logged model metrics")
            
            # Log model in MLflow
            mlflow.sklearn.log_model(model, "random_forest_model")
            
            # Use a fixed model name to increment versions
            model_name = "accident-severity-predictor"
            try:
                model_uri = f"runs:/{run.info.run_id}/random_forest_model"
                model_version = mlflow.register_model(
                    model_uri,
                    model_name
                )
                logger.info(f"Model registered in MLflow Model Registry with name: {model_name}")
                logger.info(f"Current model version: {model_version.version}, accuracy = {current_accuracy}")
                
                # Add the best_model tag
                try:
                    # First, check if there's an existing model tagged as best_model
                    best_model_version = None
                    best_model_accuracy = 0.0
                    
                    # Get all versions of the model
                    all_versions = client.search_model_versions(f"name='{model_name}'")
                    logger.info(f"Found {len(all_versions)} total versions for model {model_name}")
                    
                    # Find the version tagged as best_model
                    for version in all_versions:
                        if version.tags and "best_model" in version.tags:
                            try:
                                version_accuracy = float(version.tags["best_model"])
                                logger.info(f"Found model version {version.version} with best_model tag and accuracy {version_accuracy}")
                                if version_accuracy > best_model_accuracy:
                                    best_model_accuracy = version_accuracy
                                    best_model_version = version.version
                            except ValueError:
                                logger.warning(f"Invalid accuracy value in best_model tag: {version.tags['best_model']}")
                    
                    if best_model_version is not None:
                        logger.info(f"Current best model is version {best_model_version} with accuracy {best_model_accuracy}")
                    else:
                        logger.info("No existing model found with best_model tag")
                    
                    # Only tag the current model as best_model if it's better than previous best
                    if current_accuracy > best_model_accuracy:
                        # If we found a previous best model, remove its tag
                        if best_model_version is not None:
                            try:
                                client.delete_model_version_tag(
                                    name=model_name,
                                    version=best_model_version,
                                    key="best_model"
                                )
                                logger.info(f"Removed best_model tag from version {best_model_version}")
                            except Exception as e:
                                logger.error(f"Error removing best_model tag from version {best_model_version}: {str(e)}")
                        
                        # Tag the new model as best_model
                        try:
                            client.set_model_version_tag(
                                name=model_name,
                                version=model_version.version,
                                key="best_model",
                                value=str(current_accuracy)
                            )
                            logger.info(f"Tagged model version {model_version.version} as best_model with accuracy: {current_accuracy} (improved from {best_model_accuracy})")
                        except Exception as e:
                            logger.error(f"Error setting best_model tag for version {model_version.version}: {str(e)}")
                    else:
                        logger.info(f"Current model accuracy ({current_accuracy}) is not better than existing best model ({best_model_accuracy}), keeping existing best_model tag")
                except Exception as e:
                    logger.error(f"Error processing best_model tags: {str(e)}")
                    logger.error(traceback.format_exc())
                    # Continue even if tagging fails
                
                # Save the trained model locally
                model_path = f"{model_dir}/rf_model_{year}.joblib"
                # Always save the model at each run
                joblib.dump(model, model_path)
                logger.info(f"Model saved locally to {model_path}")
                # Only save as best_model file if it is the best
                if current_accuracy > best_model_accuracy:
                    best_model_path = f"{model_dir}/best_model_2023.joblib"
                    joblib.dump(model, best_model_path)
                    logger.info(f"Model also saved as best model to {best_model_path}")

                    # Génération du fichier de métadonnées complet
                    metadata_path = f"{model_dir}/best_model_2023_metadata.json"
                    try:
                        # Récupérer le hash du commit git
                        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
                    except Exception as e:
                        commit_hash = None
                        logger.warning(f"Could not get git commit hash: {e}")

                    # Récupérer les infos MLflow
                    experiment_id = run.info.experiment_id
                    experiment_name = experiment_name  # déjà défini plus haut
                    last_run_id = run.info.run_id
                    run_name = run.info.run_name

                    # Hyperparamètres utilisés (ceux du meilleur modèle)
                    hyperparameters = best_params.copy()
                    # On s'assure d'avoir les valeurs demandées
                    for k in ["n_estimators", "max_depth", "class_weight"]:
                        if k not in hyperparameters:
                            hyperparameters[k] = None

                    # Nombre total d'échantillons
                    total_samples = len(X)

                    # Création du dictionnaire de métadonnées
                    metadata = {
                        "model_version": str(model_version.version),
                        "model_type": model_config.get("type", "RandomForestClassifier"),
                        "accuracy": float(current_accuracy),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "git_info": {
                            "commit_hash": commit_hash
                        },
                        "mlflow_info": {
                            "experiment_id": str(experiment_id),
                            "experiment_name": experiment_name,
                            "last_run_id": last_run_id,
                            "run_name": run_name
                        },
                        "hyperparameters": {
                            "n_estimators": hyperparameters["n_estimators"],
                            "max_depth": hyperparameters["max_depth"],
                            "class_weight": hyperparameters["class_weight"]
                        },
                        "data_source": data_path,
                        "total_samples": total_samples
                    }
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)
                    logger.info(f"Full metadata written to {metadata_path}")
                    
                    # Git commit for the metadata only (not the model)
                    try:
                        # Only force-add metadata file, not the model
                        subprocess.run(["git", "add", "-f", metadata_path], check=True)
                        
                        # Using DVC to track both the model and data files
                        try:
                            # Use dvc commit instead of dvc add for model file (already defined in dvc.yaml)
                            subprocess.run(["dvc", "commit", model_path, "--force"], check=True)
                            logger.info(f"DVC commit done for model: {model_path}")
                            
                            # Check if data file is also in dvc.yaml
                            if os.path.exists("dvc.yaml"):
                                with open("dvc.yaml", "r") as f:
                                    dvc_content = f.read()
                                
                                if data_path in dvc_content:
                                    # Use commit if the file is defined in dvc.yaml
                                    subprocess.run(["dvc", "commit", data_path, "--force"], check=True)
                                    logger.info(f"DVC commit done for data: {data_path}")
                                else:
                                    # Use add if it's not defined in dvc.yaml
                                    subprocess.run(["dvc", "add", data_path], check=True)
                                    logger.info(f"DVC add done for data: {data_path}")
                            else:
                                # If no dvc.yaml, use add
                                subprocess.run(["dvc", "add", data_path], check=True)
                                logger.info(f"DVC add done for data: {data_path}")
                        except Exception as e:
                            logger.error(f"Error during DVC operations: {str(e)}")
                            logger.error(traceback.format_exc())
                        
                        # Create git commit with metadata file
                        commit_message = f"Best model version: {model_version.version}, accuracy: {current_accuracy:.4f}"
                        subprocess.run(["git", "commit", "-m", commit_message], check=True)
                        logger.info(f"Git commit done: {commit_message}")
                    except Exception as e:
                        logger.error(f"Error during git commit: {str(e)}")
                    # Promote to production (log only if best model)
                    try:
                        client.transition_model_version_stage(
                            name=model_name,
                            version=model_version.version,
                            stage="Production"
                        )
                        logger.info(f"Promoted version {model_version.version} to Production")
                    except Exception as e:
                        logger.error(f"Error promoting model to Production: {str(e)}")
                        logger.error(traceback.format_exc())
                
            except Exception as e:
                logger.error(f"Failed to register model in Model Registry: {str(e)}")
                logger.error(traceback.format_exc())
                # Continue even if model registration fails
            
            # Explicitly end the run successfully, even if registration in the registry failed
            mlflow.end_run(status="FINISHED")
            logger.info("MLflow run marked as FINISHED successfully")

            # Creating end signal file for auto_dvc
            with open("models/training.lock", "w") as f:
                f.write("done\n")

        except Exception as e:
            # In case of error, end the run with a failure status
            logger.error(f"Error during MLflow run: {str(e)}")
            logger.error(traceback.format_exc())
            mlflow.end_run(status="FAILED")
            raise

    except Exception as e:
        logger.error(f"An error occurred during model training: {str(e)}")
        logger.error(traceback.format_exc())
        # Even if there's an error, try to save a dummy model to allow the pipeline to continue
        try:
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            year = config["data_extraction"]["year"] if 'config' in locals() else "2023"
            
            # If previous model exists, use that
            previous_model_path = f"{model_dir}/rf_model_{year}.joblib"
            
            # Check if best model exists, if not create a dummy one
            best_model_path = f"{model_dir}/best_model_2023.joblib"
            if not os.path.exists(best_model_path):
                # Create a simple dummy model if needed
                dummy_model = RandomForestClassifier(n_estimators=10)
                joblib.dump(dummy_model, best_model_path)
                logger.info(f"Created fallback best model file due to error")
            
            # Create a simple dummy model if needed
            dummy_model = RandomForestClassifier(n_estimators=10)
            joblib.dump(dummy_model, previous_model_path)

            logger.info(f"Created fallback model files due to error")
        except:
            logger.error("Could not create fallback model files")
        raise

if __name__ == "__main__":
    train_model()
