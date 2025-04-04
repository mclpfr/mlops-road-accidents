from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from sklearn.metrics import classification_report
import joblib
import pandas as pd
import io

# Initialiser l'application FastAPI
app = FastAPI()

# Charger le modèle entraîné
model = joblib.load("../models/rf_model_2023.joblib")

# Endpoint pour faire des prédictions
@app.post("/predict")
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
