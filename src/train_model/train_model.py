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
import time
import traceback
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file or environment variables."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using environment variables as fallback.")
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
                "cv_folds": 5 # Not actively used in the direct training flow
            },
            "git": { # Added a section for Git configuration if needed
                "user": {
                    "name": os.getenv("GIT_USER_NAME"),
                    "email": os.getenv("GIT_USER_EMAIL")
                }
            }
        }
        if not config["mlflow"]["tracking_uri"]:
            logger.error("MLFLOW_TRACKING_URI must be set in environment variables if config.yaml is not found.")
            # sys.exit(1) # Uncomment to stop if URI is not set
        return config

def train_model(config_path="config.yaml"):
    """Main function to train the model."""
    run_id = "N/A" # Initialize run_id for potential use in error messages
    current_accuracy = 0.0 # Initialize for potential use in error messages

    try:
        config = load_config(config_path)
        year = config["data_extraction"]["year"]
        logger.info(f"Configuration loaded for year {year}")
        
        marker_file = "data/processed/prepared_data.done"
        if not os.path.exists(marker_file):
            logger.error(f"Error: Marker file {marker_file} does not exist. prepare_data.py must be run first.")
            return

        mlflow_config = config["mlflow"]
        tracking_uri = mlflow_config["tracking_uri"]
        if not tracking_uri:
            raise ValueError("MLflow tracking URI not found in config or environment variables.")
            
        logger.info(f"Setting MLflow tracking URI to: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        
        if mlflow_config.get("username") and mlflow_config.get("password"):
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_config["username"]
            os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_config["password"]
            logger.info("MLflow authentication configured.")
        else:
            logger.info("No MLflow authentication configured (username/password not provided).")

        client = mlflow.tracking.MlflowClient()
        logger.info("Successfully connected to MLflow server.")
            
        experiment_name = f"traffic-incidents-{year}" # Dynamic experiment name with year
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.info(f"Creating new experiment: {experiment_name}")
            mlflow.create_experiment(experiment_name)
        elif experiment.lifecycle_stage == "deleted":
            logger.info(f"Experiment {experiment_name} exists but is deleted, permanently deleting it.")
            client.delete_experiment(experiment.experiment_id)
            logger.info(f"Re-creating experiment: {experiment_name}") # Changed log message
            mlflow.create_experiment(experiment_name)
        else:
            logger.info(f"Using existing experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)
        logger.info(f"Experiment set to: {experiment_name}")

        data_path = f"data/processed/prepared_accidents_{year}.csv"
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)

        logger.info(f"Loading data from {data_path}")
        if not os.path.exists(data_path):
            logger.error(f"Data file {data_path} not found.")
            raise FileNotFoundError(f"Data file {data_path} not found.")
        data = pd.read_csv(data_path, low_memory=False)

        features = ["catu", "sexe", "trajet", "catr", "circ", "vosp", "prof", "plan", "surf", "situ", "lum", "atm", "col"]
        target = "grav"
        available_features = [col for col in features if col in data.columns]
        if not available_features:
            raise ValueError("None of the selected features are available in the dataset.")
        logger.info(f"Using features: {available_features}")

        X = pd.get_dummies(data[available_features], drop_first=True)
        y = data[target]
        
        if y.isna().any():
            logger.info(f"{y.isna().sum()} NaN values found in target variable. Dropping corresponding rows.")
            mask = ~y.isna()
            X = X[mask]
            y = y[mask]
            logger.info(f"After removing NaNs: {len(X)} samples remaining.")

        model_config = config.get("model", {})
        test_size = model_config.get("test_size", 0.2)
        random_state = model_config.get("random_state", 42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logger.info(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")

        with mlflow.start_run(run_name=f"run_rf_{year}_{time.strftime('%Y%m%d-%H%M%S')}") as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run started with ID: {run_id} and name: {run.info.run_name}")
            
            rf_params = {}
            if "hyperparameters" in model_config:
                for k, v_list in model_config["hyperparameters"].items():
                    if isinstance(v_list, list) and v_list:
                        rf_params[k] = v_list[0] # Takes the first element
                    elif not isinstance(v_list, list):
                         rf_params[k] = v_list # If not a list, takes the direct value
                    # else: ignore if empty list or other cases
            rf_params["random_state"] = random_state # Ensures model reproducibility
            
            model = RandomForestClassifier(**rf_params)
            logger.info(f"Training RandomForestClassifier with parameters: {rf_params}")
            model.fit(X_train, y_train)
            
            params_to_log = {
                "model_type": model_config.get("type", "RandomForestClassifier"),
                "year": year,
                "features_used": json.dumps(available_features), # Actual features used
                "test_size": test_size,
                "random_state_split": random_state, # Seed for train_test_split
                **rf_params # Model hyperparameters
            }
            mlflow.log_params(params_to_log)
            logger.info(f"Parameters logged to MLflow: {params_to_log}")
            
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            logger.info("Model evaluation completed.")
            
            current_accuracy = report["accuracy"]
            mlflow.log_metric("accuracy", current_accuracy)
            for label, metrics_dict in report.items():
                if isinstance(metrics_dict, dict):
                    for metric_name, value in metrics_dict.items():
                        mlflow.log_metric(f"{label}_{metric_name}", value)
            logger.info(f"'accuracy' metric logged: {current_accuracy:.4f}")
            
            mlflow.sklearn.log_model(model, "random_forest_model") # Artifact name within the run
            
            # Static model name in MLflow Model Registry as requested
            model_name_registry = "accident-severity-predictor"
            model_uri = f"runs:/{run_id}/random_forest_model"
            
            try:
                registered_model_version = mlflow.register_model(model_uri, model_name_registry)
                logger.info(f"Model registered in MLflow Model Registry as: {model_name_registry}, version: {registered_model_version.version}")
                
                best_model_accuracy = 0.0
                best_model_version_num = None # Stores the version number (string)
                
                # Search for versions of THIS specific model_name_registry
                for mv in client.search_model_versions(f"name='{model_name_registry}'"):
                    if mv.tags and "best_model" in mv.tags:
                        try:
                            tag_accuracy = float(mv.tags["best_model"])
                            if tag_accuracy > best_model_accuracy:
                                best_model_accuracy = tag_accuracy
                                best_model_version_num = mv.version
                        except ValueError:
                            logger.warning(f"Invalid 'best_model' tag value for version {mv.version}: {mv.tags['best_model']}")

                logger.info(f"Previous best model accuracy ({model_name_registry}, version {best_model_version_num}): {best_model_accuracy:.4f}")
                logger.info(f"Current model accuracy ({model_name_registry}, version {registered_model_version.version}): {current_accuracy:.4f}")

                best_model_filename_base = "best_model_2023" # Fixed base name for .joblib and .json files
                
                if current_accuracy > best_model_accuracy:
                    logger.info(f"Current model (acc: {current_accuracy:.4f}) is better than the previous best (acc: {best_model_accuracy:.4f}). Promoting...")
                    
                    if best_model_version_num:
                        try:
                            # Remove tag from the old best model version
                            client.delete_model_version_tag(name=model_name_registry, version=best_model_version_num, key="best_model")
                            logger.info(f"'best_model' tag removed from version {best_model_version_num} of model {model_name_registry}.")
                        except Exception as e:
                            logger.error(f"Error removing 'best_model' tag from version {best_model_version_num}: {e}")
                    
                    # Set tag for the new best model version
                    client.set_model_version_tag(name=model_name_registry, version=registered_model_version.version, key="best_model", value=str(current_accuracy))
                    logger.info(f"'best_model={current_accuracy:.4f}' tag added to version {registered_model_version.version} of model {model_name_registry}.")
                    
                    best_model_path = os.path.join(model_dir, f"{best_model_filename_base}.joblib")
                    joblib.dump(model, best_model_path)
                    logger.info(f"Model saved locally as best model: {best_model_path}")

                    metadata_path = os.path.join(model_dir, f"{best_model_filename_base}_metadata.json")
                    commit_hash = "N/A"
                    try:
                        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
                    except Exception as e:
                        logger.warning(f"Could not get Git commit hash: {e}")

                    metadata = {
                        "model_name_registry": model_name_registry, # Static name
                        "model_version": registered_model_version.version,
                        "model_type": model_config.get("type", "RandomForestClassifier"),
                        "accuracy": float(current_accuracy),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                        "git_info": {"commit_hash": commit_hash},
                        "mlflow_info": {
                            "experiment_id": run.info.experiment_id,
                            "experiment_name": client.get_experiment(run.info.experiment_id).name,
                            "run_id": run_id,
                            "run_name": run.info.run_name,
                            "model_uri": model_uri
                        },
                        "hyperparameters_used": rf_params,
                        "data_source_processed": data_path, # Processed data used for this model
                        "data_source_raw": f"data/raw/accidents_{year}_synthet.csv", # Corresponding raw data
                        "total_samples_in_X": len(X),
                        "features_definition": features # Initial list of desired features
                    }
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)
                    logger.info(f"Full metadata written to {metadata_path}")

                    # --- DVC Integration ---
                    logger.info("Starting DVC versioning for the new best model...")
                    dvc_files_to_add = [
                        data_path, # e.g., data/processed/prepared_accidents_2023.csv
                        best_model_path  # models/best_model_2023.joblib
                    ]
                    
                    raw_data_file_for_dvc = f"data/raw/accidents_{year}_synthet.csv" # Raw data file for the current year
                    # User specifically requested "data/raw/accidents_2023_synthet.csv" to be versioned.
                    # We will add this specific file if it exists.
                    # If the current year is not 2023, we might also want to version the current year's raw data.
                    
                    requested_raw_data_file = "data/raw/accidents_2023_synthet.csv"
                    if os.path.exists(requested_raw_data_file):
                        dvc_files_to_add.append(requested_raw_data_file)
                        logger.info(f"Found and will attempt to DVC add requested raw data: {requested_raw_data_file}")
                    else:
                        logger.warning(f"Requested raw data file for DVC not found: {requested_raw_data_file}. It will not be added to DVC.")

                    # Optionally, add the current year's raw data if different and exists
                    if year != "2023" and os.path.exists(raw_data_file_for_dvc) and raw_data_file_for_dvc not in dvc_files_to_add:
                        dvc_files_to_add.append(raw_data_file_for_dvc)
                        logger.info(f"Also attempting to DVC add current year's raw data: {raw_data_file_for_dvc}")
                    elif year != "2023" and not os.path.exists(raw_data_file_for_dvc):
                         logger.warning(f"Current year's raw data file ({raw_data_file_for_dvc}) not found, not adding to DVC.")


                    git_add_paths = []
                    for f_path in dvc_files_to_add:
                        if os.path.exists(f_path):
                            try:
                                logger.info(f"DVC add/commit: {f_path}")
                                # Use 'dvc commit --force' for files that are already outputs of DVC stages
                                # This will update the DVC cache without changing the pipeline definition
                                result = subprocess.run(["dvc", "commit", "--force", f_path], 
                                                     capture_output=True, text=True)
                                if result.returncode == 0:
                                    logger.info(f"Successfully committed {f_path} to DVC cache")
                                else:
                                    # If commit fails, try with add as fallback
                                    logger.info(f"File {f_path} not in DVC pipeline, trying dvc add...")
                                    result = subprocess.run(["dvc", "add", f_path], 
                                                         capture_output=True, text=True)
                                    if result.returncode == 0:
                                        git_add_paths.append(f"{f_path}.dvc") # Add the .dvc file for Git
                                    else:
                                        logger.error(f"Failed to add/commit {f_path} to DVC: {result.stderr}")
                            except subprocess.CalledProcessError as e:
                                logger.error(f"Failed to process {f_path} with DVC: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
                            except FileNotFoundError:
                                logger.error("DVC command not found. Ensure DVC is installed and in PATH.")
                                break # Stop DVC process if command is not found
                        else:
                            logger.warning(f"File {f_path} not found, skipped for 'dvc add'.")
                    
                    # --- Git Integration (continued) ---
                    if git_add_paths: # If .dvc files were created/updated
                        git_add_paths.append(metadata_path) # The metadata file itself
                        if os.path.exists(".gitignore"): git_add_paths.append(".gitignore")
                        if os.path.exists("dvc.yaml"): git_add_paths.append("dvc.yaml") # As requested

                        git_config = config.get("git", {})
                        if git_config.get("user", {}).get("name") and git_config.get("user", {}).get("email"):
                            try:
                                subprocess.run(["git", "config", "user.name", git_config["user"]["name"]], check=True)
                                subprocess.run(["git", "config", "user.email", git_config["user"]["email"]], check=True)
                                logger.info(f"Git identity configured: {git_config['user']['name']} <{git_config['user']['email']}>")
                            except Exception as e:
                                logger.warning(f"Could not configure Git user: {e}")
                        
                        for f_to_git_add in git_add_paths:
                            try:
                                logger.info(f"Git add: {f_to_git_add}")
                                # Using -f to force add if necessary (e.g., .gitignore modified by dvc, or dvc.yaml)
                                subprocess.run(["git", "add", "-f", f_to_git_add], check=True, capture_output=True, text=True)
                            except subprocess.CalledProcessError as e:
                                logger.error(f"Failed 'git add -f {f_to_git_add}': {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
                        
                        try:
                            commit_message = f"Promote best model v{registered_model_version.version} (acc: {current_accuracy:.4f}); Update DVC data/model" # English
                            logger.info(f"Git commit with message: '{commit_message}'")
                            subprocess.run(["git", "commit", "-m", commit_message], check=True, capture_output=True, text=True)
                            logger.info("Git commit successful.")
                            # Optional: git push if configured
                            # subprocess.run(["git", "push"], check=True) 
                        except subprocess.CalledProcessError as e:
                            logger.error(f"Git commit failed: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
                    else:
                        logger.warning("No .dvc files to add to Git. Git commit for DVC changes was skipped.")

                    try:
                        client.transition_model_version_stage(
                            name=model_name_registry,
                            version=registered_model_version.version,
                            stage="Production"
                        )
                        logger.info(f"Version {registered_model_version.version} of model {model_name_registry} promoted to 'Production'.")
                    except Exception as e:
                        logger.error(f"Error promoting model to 'Production': {e}\n{traceback.format_exc()}")
                else:
                    logger.info(f"Current model (acc: {current_accuracy:.4f}) is not better than previous best model (acc: {best_model_accuracy:.4f}). No promotion.")

            except Exception as e:
                logger.error(f"Failed to register model in Model Registry: {e}\n{traceback.format_exc()}")
            
            # Systematically save the current run's model locally (not necessarily the best)
            current_run_model_path = os.path.join(model_dir, f"rf_model_{year}.joblib")
            joblib.dump(model, current_run_model_path)
            logger.info(f"Current run's model saved locally: {current_run_model_path}")

            mlflow.log_artifact(current_run_model_path) # Also log this model as an artifact
            
            # MLflow run end is handled by the `with` statement

        # Create training completion marker file
        with open(os.path.join(model_dir, "train_model.done"), "w") as f:
            f.write(f"done at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\nrun_id: {run_id}\naccuracy: {current_accuracy:.4f}")
        logger.info("Marker file created: models/train_model.done")

    except FileNotFoundError as e: # Specifically handle file not found errors earlier
        logger.error(f"File not found error: {e}")
        logger.error(traceback.format_exc())
        create_fallback_models(config.get("data_extraction", {}).get("year", "2023") if 'config' in locals() else "2023")

    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        logger.error(traceback.format_exc())
        if 'run_id' in locals() and run_id != "N/A" and mlflow.active_run() and mlflow.active_run().info.run_id == run_id:
             mlflow.end_run(status="FAILED")
             logger.info(f"MLflow run {run_id} marked as FAILED.")
        create_fallback_models(config.get("data_extraction", {}).get("year", "2023") if 'config' in locals() else "2023")
        raise # Propagate the exception to indicate script failure

def create_fallback_models(year_str="2023"):
    """Creates fallback model files in case of a major error."""
    try:
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        best_model_path = os.path.join(model_dir, "best_model_2023.joblib") # Fixed name as requested
        if not os.path.exists(best_model_path):
            dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
            joblib.dump(dummy_model, best_model_path)
            logger.info(f"Fallback model file (best_model_2023.joblib) created due to an error.")

    except Exception as fallback_exc:
        logger.error(f"Could not create fallback model files: {fallback_exc}")


if __name__ == "__main__":
    config_file_path = "config.yaml" 
    
    try:
        temp_config_for_marker = load_config(config_file_path)
    except Exception as e:
        logger.error(f"Failed to load configuration for marker check: {e}")
        sys.exit(1)

    # The marker file name does not depend on the year in the original script
    data_prepared_marker_file = "data/processed/prepared_data.done" 
    
    max_wait_time = 300  # seconds
    wait_interval = 10   # seconds
    waited_time = 0
    
    logger.info(f"Waiting for marker file: {data_prepared_marker_file}")
    while not os.path.exists(data_prepared_marker_file) and waited_time < max_wait_time:
        logger.info(f"Waiting for {data_prepared_marker_file}... ({waited_time}/{max_wait_time}s)")
        time.sleep(wait_interval)
        waited_time += wait_interval
    
    if not os.path.exists(data_prepared_marker_file):
        logger.error(f"Error: Marker file {data_prepared_marker_file} was not created within the timeout period.")
        sys.exit(1) # Exit script if data is not ready
    
    logger.info(f"Marker file {data_prepared_marker_file} found. Starting model training.")
    train_model(config_path=config_file_path)

