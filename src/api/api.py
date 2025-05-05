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

# Charger le modèle entraîné
model = joblib.load("../models/rf_model_2023.joblib")

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
