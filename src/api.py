from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sklearn.metrics import classification_report
from pydantic import BaseModel
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
@app.post("/predict", current_user: str = Depends(get_current_user))
async def predict(file: UploadFile = File()):
    try:
        # Lire le fichier CSV avec pandas
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Select relevant features
        features = ["catu", "sexe", "trajet", "catr", "circ", "vosp", "prof", "plan", "surf", "situ", "lum", "atm", "col"]
        target = "grav"  # Binary target column (0: grave, 1: not grave)
        
        # Ensure all selected features exist in the dataset
        available_features = [col for col in features if col in df.columns]
        if not available_features:
            raise ValueError("None of the selected features are available in the dataset.")
        
        # Prepare features (X) and target (y)
        X = pd.get_dummies(df[available_features], drop_first=True)  # Convert categorical variables to dummy variables
        y = df[target]
        
        # Faire les prédictions avec le modèle
        y_pred = model.predict(X)
        
        # Classification report
        return JSONResponse({"classification report": classification_report(y, y_pred)})
    
    except ImportError as e:
        return JSONResponse({'error': str(e)}, status_code=500)
