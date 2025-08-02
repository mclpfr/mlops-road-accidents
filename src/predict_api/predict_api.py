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
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, ValidationError, Field
from typing import Literal
import jwt


# Essayer d'importer depuis auth_api, sinon utiliser auth_api_stub
# try:
#     from auth_api import get_current_user, User
# except ImportError:
#     try:
#         from auth_api_stub import get_current_user, User
#     except ImportError:
#         # Définir des stubs si aucun module n'est disponible
#         class User:
#             username: str
#             hashed_password: str
        
#         async def get_current_user(token: str = None):
#             return User()

# Load configuration for JWT
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
auth_config = config.get('auth_api', {})

SECRET_KEY = auth_config.get('jwt_secret_key')
ALGORITHM = auth_config.get('jwt_algorithm')

if not SECRET_KEY or not ALGORITHM:
    raise ValueError("JWT_SECRET_KEY and JWT_ALGORITHM must be set in config.yaml for predict_api")

router = APIRouter()
security = HTTPBearer()

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
    """Load configuration from a YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
    except FileNotFoundError as e:
        logger.error(f"Config file {config_path} not found.")
        raise e

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

def get_current_user_role(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise credentials_exception
        return {"username": username, "role": role}
    except jwt.PyJWTError as e:
        raise credentials_exception from e

def is_admin(user: dict = Depends(get_current_user_role)):
    print(user["role"])
    if user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return user

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

_best = find_best_model(config_path="config.yaml")
if _best is not None:
    model_use, model_version_use = _best
else:
    model_use, model_version_use = None, 0
    logger.warning("Aucun modèle trouvé lors de l'initialisation; l'API fonctionnera mais renverra une erreur si /predict est appelé avant qu'un modèle ne soit disponible.")

@router.get("/")
def verify_api():
    return {"message": "Bienvenue ! L'API prédiction est fonctionnelle."}

@router.post("/predict")
async def predict(data: InputData, payload: dict = Depends(verify_token)):
    try:
        username = payload.get("sub")
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
            "user": username,
            "prediction": y_pred.tolist(),
            "confidence": confidence
        }

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.post("/predict_csv")
async def predict_csv(file_request: UploadFile = File(), payload: dict = Depends(verify_token)):
    try:
        username = payload.get("sub")
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

        return {"user": username, "message": f"Prédiction effectuée avec succès avec le modèle version {model_version_use}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.get("/reload")
async def reload_model(user: dict = Depends(is_admin)):
    global model_use, model_version_use
    model_up, model_version_up = find_best_model(config_path="config.yaml")
    if model_version_up > model_version_use:
        model_use = model_up
        model_version_use = model_version_up
        return {"user": user["username"], "message": "Mise à jour d'un nouveau modèle effectué."}
    else:
        return {"user": user["username"], "message": "Il n'y a pas de mise à jour d'un nouveau modèle."}
