from sklearn.exceptions import UndefinedMetricWarning

import os
import io
import warnings
import logging
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, Field
from typing import Literal

# Essayer d'importer depuis auth_api, sinon utiliser auth_api_stub
try:
    from auth_api import get_current_user, User
except ImportError:
    try:
        from auth_api_stub import get_current_user, User
    except ImportError:
        # Définir des stubs si aucun module n'est disponible
        class User:
            username: str
            hashed_password: str
        
        async def get_current_user(token: str = None):
            return User()

router = APIRouter()

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modèles Pydantic pour les données
class Feature(BaseModel):
    grav: int = Field(ge=1, le=4)
    catu: int = Field(ge=1, le=3)
    sexe: int = Field(ge=1, le=2)
    trajet: Literal[-1, 0, 1, 2, 3, 4, 5, 9]
    catr: Literal[1, 2, 3, 4, 5, 6, 7, 9]
    circ: Literal[-1, 1, 2, 3, 4]
    vosp: int = Field(ge=-1, le=3)
    prof: Literal[-1, 1, 2, 3, 4]
    plan: Literal[-1, 1, 2, 3, 4]
    surf: Literal[-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    situ: Literal[-1, 0, 1, 2, 3, 4, 5, 6, 8]
    lum: int = Field(ge=1, le=5)
    atm: Literal[-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    col: Literal[-1, 1, 2, 3, 4, 5, 6, 7]

    class Config:
        extra = 'allow'

class InputData(BaseModel):
    grav: int = Field(ge=1, le=4, default=1)
    catu: int = Field(ge=1, le=3, default=1)
    sexe: int = Field(ge=1, le=2, default=1)
    trajet: Literal[-1, 0, 1, 2, 3, 4, 5, 9] = 1
    catr: Literal[1, 2, 3, 4, 5, 6, 7, 9] = 1
    circ: Literal[-1, 1, 2, 3, 4] = 1
    vosp: int = Field(ge=-1, le=3, default=1)
    prof: Literal[-1, 1, 2, 3, 4] = 1
    plan: Literal[-1, 1, 2, 3, 4] = 1
    surf: Literal[-1, 1, 2, 3, 4, 5, 6, 7, 8, 9] = 1
    situ: Literal[-1, 0, 1, 2, 3, 4, 5, 6, 8] = 1
    lum: int = Field(ge=1, le=5, default=1)
    atm: Literal[-1, 1, 2, 3, 4, 5, 6, 7, 8, 9] = 1
    col: Literal[-1, 1, 2, 3, 4, 5, 6, 7] = 1

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using environment variables")
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

def find_best_model(config_path="config.yaml"):
    try:
        config = load_config(config_path)
        year = config["data_extraction"]["year"]
        logger.info(f"Loaded configuration for year {year}")

        mlflow_config = config["mlflow"]
        tracking_uri = mlflow_config["tracking_uri"]
        if not tracking_uri:
            raise ValueError("MLflow tracking URI not found in config or environment variables")

        logger.info(f"Setting MLflow tracking URI to: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)

        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_config["username"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_config["password"]
        logger.info("MLflow authentication configured")

        try:
            client = mlflow.tracking.MlflowClient()
            logger.info("Successfully connected to MLflow server")

            registered_models = client.search_registered_models()
            model_names = [model.name for model in registered_models]
            logger.info(f"Existing registered models: {model_names}")

            for model_name in model_names:
                all_versions = client.search_model_versions(f"name='{model_name}'")
                logger.info(f"Model {model_name} has {len(all_versions)} versions")
                best_versions = [v for v in all_versions if v.tags and "best_model" in v.tags]
                if best_versions:
                    best_version = best_versions[0]
                    logger.info(f"Latest best version: {best_version.version}, Stage: {best_version.current_stage}")
                    logger.info(f"Tags: {best_version.tags}")

                    model_uri = f"models:/{model_name}/{best_version.version}"
                    model = mlflow.sklearn.load_model(model_uri)
                    model_version = {best_version.version}
                    return model, model_version

        except Exception as e:
            logger.error(f"Error checking existing models: {str(e)}")

    except Exception as e:
        logger.error(f"An error occurred during best model finding: {str(e)}")

model_use, model_version_use = find_best_model(config_path="config.yaml")

@router.post("/predict")
async def predict(data: InputData, current_user: User = Depends(get_current_user)):
    try:
        df = pd.DataFrame([data.model_dump()])
        model_features = [
            "catu",
            "sexe",
            "trajet",
            "catr",
            "circ",
            "vosp",
            "prof",
            "plan",
            "surf",
            "situ",
            "lum",
            "atm",
            "col"
        ]
        target = "grav"
        X = pd.get_dummies(df[model_features])
        y = df[target].apply(lambda x: 0 if x in [3, 4] else 1)
        # Get both prediction and probabilities
        y_pred = model_use.predict(X)
        y_proba = model_use.predict_proba(X)

        # Handle potential NaN values that would be encoded as `null` in JSON
        # and displayed as 0 % in the Streamlit front-end.
        proba_row = np.nan_to_num(y_proba[0], nan=0.0, posinf=0.0, neginf=0.0)
        confidence = float(np.max(proba_row))
        
        return {
            "user": current_user.username, 
            "prediction": y_pred.tolist(),
            "confidence": confidence
        }

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.post("/predict_csv")
async def predict_csv(file_request: UploadFile = File(), current_user: User = Depends(get_current_user)):
    try:
        contents = await file_request.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        records = df.to_dict(orient='records')
        features = [Feature(**record) for record in records]
        if not features:
            raise ValueError("None of the selected features are available in the dataset.")

        model_features = [
            "catu",
            "sexe",
            "trajet",
            "catr",
            "circ",
            "vosp",
            "prof",
            "plan",
            "surf",
            "situ",
            "lum",
            "atm",
            "col"
        ]
        target = "grav"
        X = pd.get_dummies(df[model_features], drop_first=True)
        y = df[target].apply(lambda x: 0 if x in [3, 4] else 1)
        y_pred = model_use.predict(X)

        df_ypred = pd.DataFrame(y_pred)
        df_ypred.to_csv("data/out/y_pred.csv", index=False)

        from sklearn.metrics import classification_report
        report = classification_report(y, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv("data/out/classification_report.csv")

        return {"user": current_user.username, "message": f"Prédiction effectuée avec succès avec le modèle version {model_version_use}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.get("/reload")
async def reload_model(current_user: User = Depends(get_current_user)):
    global model_use, model_version_use
    model_up, model_version_up = find_best_model(config_path="config.yaml")
    if model_version_up > model_version_use:
        model_use = model_up
        model_version_use = model_version_up
        return {"user": current_user.username, "message": "Mise à jour d'un nouveau modèle effectué."}
    else:
        return {"user": current_user.username, "message": "Il n'y a pas de mise à jour d'un nouveau modèle."}
