from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sklearn.metrics import classification_report
from pydantic import BaseModel, ValidationError
import joblib
import pandas as pd
import io

# Initialiser l'application FastAPI
app = FastAPI()
security = HTTPBasic()

# Identification des utilisateurs pour faire des tests
users = {
    "alice": "wonder",
    "clem": "juice"
}

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
        model_features = ["catu", "sexe", "trajet", "catr", "circ", "vosp", "prof", "plan", "surf", "situ", "lum", "atm", "col"]
        target = "grav"
        
        # Prepare features (X) and target (y)
        X = pd.get_dummies(df[model_features], drop_first=True)  # Convert categorical variables to dummy variables
        y = df[target].apply(lambda x: 0 if x in [3, 4] else 1)  # Binary target column (0: grave, 1: not grave)
        
        # Faire les prédictions avec le modèle
        y_pred = model.predict(X)
        
        # Sortie des résultats
        df_ypred = pd.DataFrame(y_pred)
        df_ypred.to_csv("../data/out/y_pred.csv", index=False)
        
        report = classification_report(y, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv("../data/out/classification_report.csv")
        
        return {"message": "Prédiction effectuée avec succès"}
        
        # Classification report
        #return JSONResponse({"classification report": classification_report(y, y_pred)})
        #return {"classification report": classification_report(y, y_pred, output_dict=True)}
    
    except ImportError as e:
        return JSONResponse({'error': str(e)}, status_code=500)
    except ValidationError as e:
        # Retourner une erreur si la validation échoue
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        # Gérer d'autres exceptions possibles
        raise HTTPException(status_code=500, detail=str(e))