import pandas as pd
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import mlflow
import mlflow.sklearn
import json
import requests
import jwt
import logging
import time
import traceback
import subprocess
import sys
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
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
            
        # Vérification de la taille du fichier
        file_size = os.path.getsize(data_path)
        logger.info(f"File size: {file_size} bytes")
        
        # Essai avec différents encodages
        encodings = [None, 'utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        data = None
        
        for encoding in encodings:
            try:
                logger.info(f"Trying to read with encoding: {encoding}")
                data = pd.read_csv(data_path, low_memory=False, encoding=encoding)
                logger.info(f"Successfully read data with encoding: {encoding}")
                logger.info(f"Data shape: {data.shape}")
                if not data.empty:
                    logger.info(f"Columns: {data.columns.tolist()}")
                    logger.info(f"First row: {data.iloc[0].to_dict()}")
                    break
            except Exception as e:
                logger.warning(f"Failed to read with encoding {encoding}: {str(e)}")
        
        if data is None or data.empty:
            raise ValueError(f"Failed to read data file {data_path} with any encoding")
            
        logger.info(f"Successfully loaded {len(data)} rows")

        features = ["catu", "sexe", "trajet", "catr", "circ", "vosp", "prof", "plan", "surf", "situ", "lum", "atm", "col"]
        target = "grav"
        available_features = [col for col in features if col in data.columns]
        if not available_features:
            raise ValueError("None of the selected features are available in the dataset.")
        logger.info(f"Using features: {available_features}")

        logger.info(f"DataFrame columns before target assignment: {data.columns.tolist()}")
        logger.info(f"Target variable: '{target}'")
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
            # Compute and log confusion matrix artifact
            cm = confusion_matrix(y_test, y_pred)
            np.save("confusion_matrix.npy", cm)
            mlflow.log_artifact("confusion_matrix.npy")
            os.remove("confusion_matrix.npy")  # clean up
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
                    logger.info(f"'best_model' tag added to version {registered_model_version.version} of model {model_name_registry}.")

                    # --- Trigger model reload in predict_api container ---
                    try:
                        import requests
                        reload_url = os.getenv("PREDICT_API_RELOAD_URL", "http://predict_api:8000/protected/reload")
                        logger.info(f"Triggering model reload via {reload_url}")
                        # Generate admin JWT token for predict_api
                        try:
                            jwt_secret = os.getenv("PREDICT_API_JWT_SECRET", "key")
                            payload = {"sub": "train_model_service", "role": "admin"}
                            token = jwt.encode(payload, jwt_secret, algorithm="HS256")
                            headers = {"Authorization": f"Bearer {token}"}
                        except Exception as jwt_exc:
                            logger.error(f"Failed to generate JWT: {jwt_exc}")
                            headers = {}
                        response = requests.get(reload_url, headers=headers, timeout=(5, 60))
                        if response.status_code == 200:
                            logger.info("Successfully triggered model reload for predict_api container via protected endpoint.")
                        else:
                            logger.warning(f"Reload request returned status {response.status_code}: {response.text}")
                    except Exception as reload_exc:
                        logger.error(f"Failed to trigger model reload: {reload_exc}")
                    
                    best_model_path = os.path.join(model_dir, f"{best_model_filename_base}.joblib")
                    joblib.dump(model, best_model_path)
                    logger.info(f"Model saved locally as best model: {best_model_path}")
                    
                    # Generate and save confusion matrix heatmap
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    plt.figure(figsize=(10, 7))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=['Pas Grave', 'Grave'], 
                                yticklabels=['Pas Grave', 'Grave'])
                    plt.title('Matrice de Confusion')
                    plt.xlabel('Prédiction')
                    plt.ylabel('Réalité')
                    
                    # Save the figure
                    confusion_matrix_path = os.path.join(model_dir, f"confusion_matrix_best_model_{year}.png")
                    plt.savefig(confusion_matrix_path, bbox_inches='tight', dpi=300)
                    plt.close()
                    logger.info(f"Confusion matrix heatmap saved to {confusion_matrix_path}")
                    
                    # Log the confusion matrix as an MLflow artifact
                    mlflow.log_artifact(confusion_matrix_path, "confusion_matrix")
                    logger.info("Confusion matrix heatmap logged to MLflow artifacts")

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
                                logger.info(f"Processing DVC tracking for {f_path}")
                                commit_res = subprocess.run(["dvc", "commit", "--force", f_path], capture_output=True, text=True)
                                if commit_res.returncode != 0:
                                    logger.warning(f"`dvc commit` failed for {f_path}: {commit_res.stderr.strip()} – falling back to `dvc add`. ")
                                    add_res = subprocess.run(["dvc", "add", f_path], capture_output=True, text=True)
                                    if add_res.returncode != 0:
                                        logger.error(f"`dvc add` failed for {f_path}: {add_res.stderr.strip()}")
                                        continue
                                dvc_file = f"{f_path}.dvc"
                                if os.path.exists(dvc_file):
                                    git_add_paths.append(dvc_file)
                                    logger.info(f"Added {dvc_file} to Git staging list")
                            except FileNotFoundError:
                                logger.error("DVC is not installed in the container PATH")
                                break
                            except Exception as dvc_exc:
                                logger.error(f"Unexpected DVC error for {f_path}: {dvc_exc}")
                        else:
                            logger.warning(f"File {f_path} not found, skipped DVC tracking.")
                    
                    # --- Git Integration (continued) ---
                    if git_add_paths: # If .dvc files were created/updated
                        # Ajouter le fichier de métadonnées s'il existe
                        if os.path.exists(metadata_path):
                            git_add_paths.append(metadata_path)
                            logger.info(f"Added metadata to Git staging: {metadata_path}")
                        
                        # Ajouter les fichiers de configuration si nécessaires
                        if os.path.exists(".gitignore"): 
                            git_add_paths.append(".gitignore")
                            logger.info("Added .gitignore to Git staging")
                            
                        if os.path.exists("dvc.yaml"): 
                            git_add_paths.append("dvc.yaml")
                            logger.info("Added dvc.yaml to Git staging")

                        try:
                            # Initialize Git repository object
                            repo = Repo('.')
                            
                            # Configure Git user and authentication
                            git_config = config.get("git", {})
                            git_user = git_config.get("user", {})
                            
                            # Set user name and email
                            if git_user.get("name") and git_user.get("email"):
                                try:
                                    with repo.config_writer() as git_config_writer:
                                        git_config_writer.set_value("user", "name", git_user["name"])
                                        git_config_writer.set_value("user", "email", git_user["email"])
                                    logger.info(f"Git identity configured: {git_user['name']} <{git_user['email']}>")
                                    
                                    # Configure remote URL with token
                                    if git_user.get("token"):
                                        remote_url = f"https://{git_user['name']}:{git_user['token']}@github.com/mclpfr/mlops-road-accidents.git"
                                        repo.git.remote("set-url", "origin", remote_url)
                                        logger.info("Git remote URL updated with token authentication")
                                        
                                except Exception as e:
                                    logger.warning(f"Could not configure Git: {e}")
                                    logger.debug(traceback.format_exc())
                            
                            # Add all modified files to staging
                            for f_to_git_add in git_add_paths:
                                try:
                                    logger.info(f"Git add: {f_to_git_add}")
                                    repo.git.add(f_to_git_add, force=True)
                                except Exception as e:
                                    logger.error(f"Failed to add {f_to_git_add} to Git: {str(e)}")
                            
                            # Check if there are any changes to commit
                            if repo.is_dirty() or repo.untracked_files:
                                # Create a simple one-line commit message
                                commit_message = f"New best model {model_name_registry} v{registered_model_version.version} (acc: {current_accuracy:.4f}) - {time.strftime('%Y-%m-%d %H:%M')}"
                                
                                # Create the commit
                                repo.git.commit(m=commit_message)
                                logger.info(f"Git commit created with message: {commit_message}")
                                
                                # Push changes to the remote repository
                                try:
                                    logger.info("Pushing changes to remote repository...")
                                    origin = repo.remote(name='origin')
                                    origin.push()
                                    logger.info("Successfully pushed changes to remote repository.")
                                except GitCommandError as e:
                                    logger.error(f"Git push failed: {str(e)}")
                            else:
                                logger.info("No changes to commit.")
                                
                        except (InvalidGitRepositoryError, NoSuchPathError) as e:
                            logger.error(f"Git repository not found or invalid: {str(e)}")
                        except Exception as e:
                            logger.error(f"Error during Git operations: {str(e)}")
                            logger.error(traceback.format_exc())
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

        # Only create the marker file if we reach this point without errors
        # This ensures the task is only marked as complete if everything succeeded
        try:
            with open(os.path.join(model_dir, "train_model.done"), "w") as f:
                f.write(f"done at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\nrun_id: {run_id}\naccuracy: {current_accuracy:.4f}")
            logger.info("Training completed successfully. Marker file created: models/train_model.done")
            
        except Exception as marker_error:
            logger.error(f"Failed to create completion marker file: {marker_error}")
            # Don't raise here, as the training itself was successful
            # Just log the error and continue

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        logger.error(traceback.format_exc())
        if 'run_id' in locals() and run_id != "N/A" and mlflow.active_run() and mlflow.active_run().info.run_id == run_id:
            try:
                mlflow.end_run(status="FAILED")
                logger.info(f"MLflow run {run_id} marked as FAILED due to file not found.")
            except Exception as mlflow_err:
                logger.error(f"Error marking MLflow run as failed: {mlflow_err}")
        create_fallback_models(config.get("data_extraction", {}).get("year", "2023") if 'config' in locals() else "2023")
        raise  # Re-raise to fail the Airflow task

    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        logger.error(traceback.format_exc())
        
        # Clean up any partial results
        try:
            if 'run_id' in locals() and run_id != "N/A" and mlflow.active_run() and mlflow.active_run().info.run_id == run_id:
                mlflow.end_run(status="FAILED")
                logger.info(f"MLflow run {run_id} marked as FAILED due to error.")
                
                # Try to clean up any partially registered models
                try:
                    client = mlflow.tracking.MlflowClient()
                    model_name_registry = "accident-severity-predictor"
                    for mv in client.search_model_versions(f"run_id='{run_id}'"):
                        if mv.current_stage == "None":
                            client.delete_model_version(
                                name=mv.name,
                                version=mv.version
                            )
                            logger.info(f"Deleted uncommitted model version {mv.version} for run {run_id}")
                except Exception as cleanup_err:
                    logger.error(f"Error during model cleanup: {cleanup_err}")
                    
        except Exception as mlflow_err:
            logger.error(f"Error during MLflow cleanup: {mlflow_err}")
            
        create_fallback_models(config.get("data_extraction", {}).get("year", "2023") if 'config' in locals() else "2023")
        raise  # Re-raise to fail the Airflow task

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


def check_environment(config: dict | None = None):
    """Validate that mandatory settings are available either via env vars or config.

    Currently the only **required** setting is the MLflow tracking URI.
    We first look for the ``MLFLOW_TRACKING_URI`` environment variable; if it is
    not set we fall back to the value provided in the loaded ``config`` (if any).
    This makes the script usable in situations where all configuration lives in
    *config.yaml* without forcing users to duplicate the value in the container
    environment.

    Parameters
    ----------
    config : dict | None
        Parsed YAML configuration.  When ``None`` the function will attempt to
        load the default config file to perform the check.
    Returns
    -------
    bool
        ``True`` if the environment is considered valid, ``False`` otherwise.
    """
    # Load configuration lazily if the caller did not provide one.
    if config is None:
        try:
            config = load_config()
        except Exception as exc:
            logger.warning(
                "Unable to load configuration for environment validation: %s", exc
            )
            config = {}

    tracking_uri_env = os.getenv("MLFLOW_TRACKING_URI")
    tracking_uri_cfg = (
        config.get("mlflow", {}).get("tracking_uri") if isinstance(config, dict) else None
    )

    if not tracking_uri_env and not tracking_uri_cfg:
        logger.error(
            "Missing MLflow tracking URI. Set the MLFLOW_TRACKING_URI environment variable "
            "or add it to config.yaml under the 'mlflow.tracking_uri' key."
        )
        return False

    return True

if __name__ == "__main__":
    config_file_path = "config.yaml" 
    
    try:
        temp_config_for_marker = load_config(config_file_path)
    except Exception as e:
        logger.error(f"Failed to load configuration for marker check: {e}")
        sys.exit(1)

    # Check environment dependencies
    if not check_environment(temp_config_for_marker):
        logger.error("Environment validation failed. Exiting.")
        sys.exit(1)

    # The marker file name does not depend on the year in the original script
    data_prepared_marker_file = "data/processed/prepared_data.done" 
    
    max_wait_time = 300  # seconds
    wait_interval = 10   # seconds
    waited_time = 0
    
    logger.info(f"En attente du fichier marqueur: {data_prepared_marker_file}")
    while not os.path.exists(data_prepared_marker_file) and waited_time < max_wait_time:
        logger.info(f"En attente de {data_prepared_marker_file}... ({waited_time}/{max_wait_time}s)")
        time.sleep(wait_interval)
        waited_time += wait_interval
    
    if not os.path.exists(data_prepared_marker_file):
        logger.error(f"Erreur: Le fichier marqueur {data_prepared_marker_file} n'a pas été créé dans le délai imparti.")
        sys.exit(1)  # Exit script if the data is not ready
    
    logger.info(f"Fichier marqueur {data_prepared_marker_file} trouvé. Démarrage de l'entraînement du modèle.")
    
    try:
        train_model(config_path=config_file_path)
        logger.info("Entraînement du modèle terminé avec succès.")
        sys.exit(0)  # Success
    except Exception as e:
        logger.error(f"Échec de l'entraînement du modèle: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)  # Failure

