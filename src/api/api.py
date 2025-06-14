import os
import io
import warnings
import logging
from datetime import datetime, timedelta, timezone
from typing import Literal
import yaml
# import json
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
from pydantic import BaseModel, ValidationError, Field

# Addition to detect Streamlit environment
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Initialize FastAPI application
app = FastAPI()
security = HTTPBasic()

# Security configuration
JWT_SECRET_KEY = "key"
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Load sensitive information from JSON file
# with open('.config.json', encoding='utf-8') as f:
#     users = json.load(f)

# Pydantic models for data
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

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    hashed_password: str

# Example
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "hashed_password": pwd_context.hash("johnsecret")
    }
}

# OAuth2 schema for token retrieval
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_mlflow_config():
    """Retrieves MLflow configuration from Streamlit secrets or environment variables"""
    config = {}
    
    # Try Streamlit secrets first if available
    if STREAMLIT_AVAILABLE:
        try:
            if hasattr(st, 'secrets') and "mlflow" in st.secrets:
                logger.info("MLflow configuration found in Streamlit secrets")
                mlflow_secrets = st.secrets["mlflow"]
                config = {
                    "tracking_uri": mlflow_secrets.get("tracking_uri"),
                    "username": mlflow_secrets.get("username", ""),
                    "password": mlflow_secrets.get("password", "")
                }
                return config
        except Exception as e:
            logger.warning(f"Error reading Streamlit secrets: {e}")
    
    # Fallback to environment variables
    logger.info("Using environment variables for MLflow")
    config = {
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
        "username": os.getenv("MLFLOW_TRACKING_USERNAME", ""),
        "password": os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    }
    
    return config

def load_config(config_path="config.yaml"):
    try:
        # Load the configuration file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            
            # Override MLflow config with Streamlit secrets if available
            mlflow_config = get_mlflow_config()
            if mlflow_config.get("tracking_uri"):
                config["mlflow"] = mlflow_config
                logger.info("MLflow configuration overridden with Streamlit secrets/env vars")
            
            return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using environment variables")
        
        # Get MLflow configuration from Streamlit secrets or env vars
        mlflow_config = get_mlflow_config()
        
        # Check that tracking URI is defined
        if not mlflow_config.get("tracking_uri"):
            # Default URI for local MLflow
            mlflow_config["tracking_uri"] = "http://localhost:5000"
            logger.warning(f"No MLflow URI found, using default: {mlflow_config['tracking_uri']}")
        
        # Fallback to environment variables
        config = {
            "data_extraction": {
                "year": os.getenv("DATA_YEAR", "2023"),
                "url": os.getenv("DATA_URL", "https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/")
            },
            "mlflow": mlflow_config
        }
        return config

# Download the best model locally
def find_best_model(config_path="config.yaml"):
    try:
        # Load configuration parameters
        config = load_config(config_path)
        year = config["data_extraction"]["year"]
        logger.info(f"Loaded configuration for year {year}")

        # Configure MLflow from config.yaml, Streamlit secrets, or environment variables
        mlflow_config = config["mlflow"]
        tracking_uri = mlflow_config["tracking_uri"]
        
        if not tracking_uri:
            raise ValueError("MLflow tracking URI not found in config, Streamlit secrets, or environment variables")

        logger.info(f"Setting MLflow tracking URI to: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)

        # Configure authentication if provided
        username = mlflow_config.get("username")
        password = mlflow_config.get("password")
        
        if username and password:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password
            logger.info("MLflow authentication configured")
        else:
            logger.info("No MLflow authentication credentials provided")

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

                        # Download the model version with "best_model" tag
                        model_uri = f"models:/{model_name}/{best_version.version}"
                        
                        # Load the model
                        model = mlflow.sklearn.load_model(model_uri)
                        model_version = {best_version.version}
                        logger.info(f"Successfully loaded model {model_name} version {best_version.version}")
                        return model, model_version

                # If no model with "best_model" tag is found
                logger.warning("No model with 'best_model' tag found")
                return None, None

            except Exception as e:
                logger.error(f"Error checking existing models: {str(e)}")
                return None, None

        except Exception as e:
            logger.error(f"Failed to connect to MLflow server: {str(e)}")
            # In case of connection failure, return None rather than raise exception
            return None, None

    except Exception as e:
        logger.error(f"An error occurred during best model finding: {str(e)}")
        return None, None

# Function to verify password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Function to get a user
def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return User(**user_dict)

# Function to authenticate a user
def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

# Function to create an access token
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

# Model loading with error handling
try:
    model_use, model_version_use = find_best_model(config_path="config.yaml")
    if model_use is None:
        logger.warning("No model loaded from MLflow, API will work in limited mode")
        model_version_use = None
    else:
        logger.info(f"Model loaded successfully, version: {model_version_use}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model_use = None
    model_version_use = None

# Endpoint to check the API
@app.get("/")
def verify_api():
    model_status = "Model loaded" if model_use is not None else "No model available"
    return {
        "message": "Bienvenue ! L'API est fonctionnelle.",
        "model_status": model_status,
        "model_version": model_version_use
    }

# Endpoint to get a token
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Function to get current user
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=username)
    if user is None:
        raise credentials_exception
    return user

# Secure endpoint
@app.get("/protected/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# Endpoint for predictions
@app.post("/protected/predict")
async def predict(data: InputData, current_user: User = Depends(get_current_user)):
    # Check that model is available
    if model_use is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not available. Please check MLflow configuration."
        )
    
    try:
        # Input data
        df = pd.DataFrame([data.model_dump()])

        # Model features selection
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

        # Prepare features (X) and target (y)
        X = pd.get_dummies(df[model_features])
        y = df[target].apply(lambda x: 0 if x in [3, 4] else 1)  # 0: grave, 1: not grave

        # Predictions with the model
        y_pred = model_use.predict(X)

        # Output results
        return {
            "user": current_user, 
            "prediction": y_pred.tolist(),
            "model_version": model_version_use
        }

    except ValidationError as e:
        # Return error if validation fails
        raise HTTPException(status_code=422, detail=e.errors()) from e
    except Exception as e:
        # Handle other possible exceptions
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/protected/predict_csv")
async def predict_csv(file_request: UploadFile = File(), current_user: User = Depends(get_current_user)):
    # Vérifier que le modèle est disponible
    if model_use is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not available. Please check MLflow configuration."
        )
    
    try:
        # Lire le fichier CSV avec pandas
        contents = await file_request.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        records = df.to_dict(orient='records')

        # Validation de chaque enregistrement avec le modèle Pydantic
        features = [Feature(**record) for record in records]
        if not features:
            raise ValueError("None of the selected features are available in the dataset.")

        # Sélection model features
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

        # Préparation features (X) et target (y)
        X = pd.get_dummies(df[model_features], drop_first=True)
        y = df[target].apply(lambda x: 0 if x in [3, 4] else 1)  # 0: grave, 1: not grave

        # Prédictions avec le modèle
        y_pred = model_use.predict(X)

        # Sortie des résultats
        df_ypred = pd.DataFrame(y_pred)
        df_ypred.to_csv("data/out/y_pred.csv", index=False)

        report = classification_report(y, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv("data/out/classification_report.csv")

        return {
            "user": current_user, 
            "message": f"Prédiction effectuée avec succès avec le modèle version {model_version_use}",
            "model_version": model_version_use
        }

    except ImportError as e:
        return JSONResponse({'error': str(e)}, status_code=500)
    except ValidationError as e:
        # Retourner une erreur si la validation échoue
        raise HTTPException(status_code=422, detail=e.errors()) from e
    except Exception as e:
        # Gérer d'autres exceptions possibles
        raise HTTPException(status_code=500, detail=str(e)) from e

# Endpoint pour mettre à jour le modèle
@app.get("/protected/reload")
async def reload_model(current_user: User = Depends(get_current_user)):
    global model_use, model_version_use
    
    try:
        model_up, model_version_up = find_best_model(config_path="config.yaml")
        
        if model_up is None:
            return {
                "user": current_user, 
                "message": "Aucun modèle disponible dans MLflow."
            }
        
        if model_version_up and model_version_use and model_version_up > model_version_use:
            model_use = model_up
            model_version_use = model_version_up
            return {
                "user": current_user, 
                "message": f"Mise à jour effectuée vers le modèle version {model_version_up}."
            }
        else:
            return {
                "user": current_user, 
                "message": "Aucune mise à jour de modèle disponible.",
                "current_version": model_version_use
            }
    except Exception as e:
        logger.error(f"Error during model reload: {e}")
        return {
            "user": current_user, 
            "message": f"Erreur lors du rechargement du modèle : {str(e)}"
        }

# Lancer l'application avec Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
