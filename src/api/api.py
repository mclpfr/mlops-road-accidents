import io
import json
import pandas as pd
import joblib
#import mlflow
#from mlflow.tracking import MlflowClient
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sklearn.metrics import classification_report
from pydantic import BaseModel, ValidationError

# Initialiser l'application FastAPI
app = FastAPI()
security = HTTPBasic()

# Charger les informations sensibles depuis le fichier JSON
with open('.config.json', encoding='utf-8') as f:
    users = json.load(f)

# Modèles Pydantic pour les données
class Feature(BaseModel):
    catu: int
    sexe: int
    trajet: int
    catr: int
    circ: int
    vosp: int
    prof: int
    plan: int
    surf: int
    situ: int
    lum: int
    atm: int
    col: int

    class Config:
        extra = 'allow'

# Configurer MLflow pour pointer vers votre serveur de suivi
#mlflow.set_tracking_uri("https://dagshub.com/mclpfr/mlops-road-accidents.mlflow")

# Initialiser le client MLflow
#client = MlflowClient()

# Rechercher le modèle enregistré avec le tag "best_model"
#registered_models = client.search_registered_models(filter_string="tag.best_model='true'")

# Vérifier si des modèles ont été trouvés
#if not registered_models:
#    raise ValueError("Aucun modèle enregistré trouvé avec le tag 'best_model'.")

# Supposons que nous prenons le premier modèle trouvé
#best_model = registered_models[0]

# Télécharger la dernière version du modèle
#model_name = best_model.name
#model_version = best_model.latest_versions[0].version
#model_uri = f"models:/{model_name}/{model_version}"

# Charger le modèle
#model = mlflow.pyfunc.load_model(model_uri)

# Télécharger le meilleur modèle localement
def best_model(config_path="config.yaml"):
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

# Charger le modèle entraîné
model = joblib.load("../models/best_model_2023.joblib")

# Fonction pour vérifier l'authentification
def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = users.get(credentials.username)
    if not (correct_username and credentials.password == correct_username):
        raise HTTPException(status_code=401, detail="Identifiants incorrects")
    return credentials.username

# Endpoint pour vérifier l'API
@app.get("/verify")
def verify_api():
    return {"message": "L'API est fonctionnelle."}

# Endpoint pour faire des prédictions
@app.post("/predict")
async def predict_csv(file_request: UploadFile = File(), current_user: str = Depends(get_current_user)):
    try:
        # Lire le fichier CSV avec pandas
        contents = await file_request.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        records = df.to_dict(orient='records')

        # Valider chaque enregistrement avec le modèle Pydantic
        features = [Feature(**record) for record in records]
        if not features:
            raise ValueError("None of the selected features are available in the dataset.")

        #return {"message": "Données validées avec succès", "features": features}

        # Select model features
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
        X = pd.get_dummies(df[model_features], drop_first=True)
        y = df[target].apply(lambda x: 0 if x in [3, 4] else 1)  # 0: grave, 1: not grave

        # Faire les prédictions avec le modèle
        y_pred = model.predict(X)

        # Sortie des résultats
        df_ypred = pd.DataFrame(y_pred)
        df_ypred.to_csv("../data/out/y_pred.csv", index=False)

        report = classification_report(y, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv("../data/out/classification_report.csv")

        return {"user": current_user, "message": "Prédiction effectuée avec succès"}

        # Classification report
        #return JSONResponse({"classification report": classification_report(y, y_pred)})
        #return {"classification report": classification_report(y, y_pred, output_dict=True)}

    except ImportError as e:
        return JSONResponse({'error': str(e)}, status_code=500)
    except ValidationError as e:
        # Retourner une erreur si la validation échoue
        raise HTTPException(status_code=422, detail=e.errors()) from e
    except Exception as e:
        # Gérer d'autres exceptions possibles
        raise HTTPException(status_code=500, detail=str(e)) from e

# Endpoint pour mettre à jour le modèle
@app.get("/reload")
async def reload_model(current_user: str = Depends(get_current_user)):
    model = joblib.load("../models/best_model_2023.joblib")
    joblib.dump(model, "best_model_2023.joblib")

    return {"message": "Modèle rechargé avec succès"}
