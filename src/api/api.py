""" API """

import os
import io
import yaml
# import json
import pandas as pd
import warnings
import logging
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
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Initialiser l'application FastAPI
app = FastAPI()
security = HTTPBasic()

# Configuration de la sécurité
JWT_SECRET_KEY = "key"
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Context de hachage des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Charger les informations sensibles depuis le fichier JSON
# with open('.config.json', encoding='utf-8') as f:
#     users = json.load(f)

# Modèles Pydantic pour les données
class Feature(BaseModel):
    grav: int = Field(ge=1, le=4)
    catu: int = Field(ge=1, le=3)
    sexe: int = Field(ge=1, le=2)
    trajet: int = Field(ge=0, le=9)
    catr: int = Field(ge=1, le=9)
    circ: int = Field(ge=1, le=4)
    vosp: int = Field(ge=0, le=3)
    prof: int = Field(ge=1, le=4)
    plan: int = Field(ge=1, le=4)
    surf: int = Field(ge=1, le=9)
    situ: int = Field(ge=0, le=8)
    lum: int = Field(ge=1, le=5)
    atm: int = Field(ge=1, le=9)
    col: int = Field(ge=1, le=7)

    class Config:
        extra = 'allow'

class InputData(BaseModel):
    grav: int = 1
    catu: int = 1
    sexe: int = 1
    trajet: int = 1
    catr: int = 1
    circ: int = 1
    vosp: int = 1
    prof: int = 1
    plan: int = 1
    surf: int = 1
    situ: int = 1
    lum: int = 1
    atm: int = 1
    col: int = 1

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    hashed_password: str

# Exemple
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "hashed_password": pwd_context.hash("johnsecret")
    }
}

# OAuth2 schema pour la récupération du token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Fonction pour vérifier l'authentification
# def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
#     correct_username = users.get(credentials.username)
#     if not (correct_username and credentials.password == correct_username):
#         raise HTTPException(status_code=401, detail="Identifiants incorrects")
#     return credentials.username

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
def find_best_model(config_path="config.yaml"):
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
                        model_version = {best_version.version}
                        return model, model_version

            except Exception as e:
                logger.error(f"Error checking existing models: {str(e)}")

        except Exception as e:
            logger.error(f"Failed to connect to MLflow server: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"An error occurred during best model finding: {str(e)}")

# Fonction pour vérifier le mot de passe
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Fonction pour obtenir un utilisateur
def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return User(**user_dict)

# Fonction pour authentifier un utilisateur
def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

# Fonction pour créer un token d'accès
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

# Chargement du modèle
model_use, model_version_use = find_best_model(config_path="config.yaml")

# Endpoint pour vérifier l'API
@app.get("/")
def verify_api():
    return {"message": "Bienvenue ! L'API est fonctionnelle."}

# Endpoint pour obtenir un token
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

# Fonction pour obtenir l'utilisateur courant
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

# Endpoint sécurisé
@app.get("/protected/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# Endpoint pour faire des prédictions
@app.post("/protected/predict")
async def predict(data: InputData, current_user: User = Depends(get_current_user)):
    try:
        # Données en entrée
        df = pd.DataFrame([data.model_dump()])

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
        X = pd.get_dummies(df[model_features])
        y = df[target].apply(lambda x: 0 if x in [3, 4] else 1)  # 0: grave, 1: not grave

        # Prédictions avec le modèle
        y_pred = model_use.predict(X)

        # Sortie des résultats
        return {"user": current_user, "prediction": y_pred.tolist()}

    except ValidationError as e:
        # Retourner une erreur si la validation échoue
        raise HTTPException(status_code=422, detail=e.errors()) from e
    except Exception as e:
        # Gérer d'autres exceptions possibles
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/protected/predict_csv")
async def predict_csv(file_request: UploadFile = File(), current_user: User = Depends(get_current_user)):
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

        return {"user": current_user, "message": f"Prédiction effectuée avec succès avec le modèle version {model_version_use}"}

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
    global model_use
    model_up, model_version_up = find_best_model(config_path="config.yaml")
    if model_version_up > model_version_use:
        model_use = model_up
        return {"user": current_user, "message": "Mise à jour d'un nouveau modèle effectué."}
    else:
        return {"user": current_user, "message": "Il n'y a pas de mise à jour d'un nouveau modèle."}

# Lancer l'application avec Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
