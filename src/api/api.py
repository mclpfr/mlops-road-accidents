import io
import json
import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sklearn.metrics import classification_report
from pydantic import BaseModel, ValidationError
from datetime import datetime, timedelta, timezone

# Initialiser l'application FastAPI
app = FastAPI()
security = HTTPBasic()

# Configuration de la sécurité
SECRET_KEY = "key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Context de hachage des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Charger les informations sensibles depuis le fichier JSON
# with open('.config.json', encoding='utf-8') as f:
#     users = json.load(f)

# Modèles Pydantic pour les données
class Feature(BaseModel):
    # Colonnes exactes attendues par le modèle
    catu: float
    sexe: float
    trajet: float
    catr: float
    circ: float
    vosp: float
    prof: float
    plan: float
    surf: float
    situ: float
    lum: float
    atm: float
    col: float

    class Config:
        extra = 'allow'

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    hashed_password: str

# class UserCreate(BaseModel):
#     username: str
#     password: str

# Charger le modèle entraîné et le scaler
model_dir = "models"
model = joblib.load(f"{model_dir}/best_model_2023.joblib")
print(f"--- TYPE DU MODÈLE CHARGÉ ---")
print(type(model))
print(f"--- PARAMÈTRES DU MODÈLE CHARGÉ (get_params) ---")
try:
    print(model.get_params(deep=False))
except AttributeError:
    print("Le modèle n'a pas de méthode get_params.")
print(f"--- FIN INFOS MODÈLE ---")

# Charger le scaler pour normaliser les données
try:
    scaler = joblib.load("data/processed/scaler_model_features_2023.joblib")
    with open("data/processed/numerical_columns_scaled_2023.json", 'r') as f:
        columns_to_scale = json.load(f)
    print(f"--- COLUMNS_TO_SCALE AU DÉMARRAGE DE L'API ---")
    print(columns_to_scale)
    print(f"--- FIN COLUMNS_TO_SCALE AU DÉMARRAGE ---")
    print("Scaler et colonnes à normaliser chargés avec succès")
except FileNotFoundError:
    print(f"Erreur lors du chargement du scaler: FileNotFoundError")
    scaler = None
    columns_to_scale = []

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
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

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
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
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
@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# Endpoint pour faire des prédictions
@app.post("/protected/predict")
async def predict_csv(file_request: UploadFile = File(), current_user: User = Depends(get_current_user)):
    try:
        print("\n--- Début du traitement predict_csv ---")
        # Lire le fichier CSV avec pandas
        contents = await file_request.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        print(f"CSV chargé. Nombre de lignes: {len(df)}")
        print(f"Colonnes initiales du CSV: {df.columns.tolist()}")

        records = df.to_dict(orient='records')

        # Valider chaque enregistrement avec le modèle Pydantic
        features = [Feature(**record) for record in records]
        if not features:
            raise ValueError("None of the selected features are available in the dataset.")

        # Colonnes attendues par le modèle
        model_features = ["catu", "sexe", "trajet", "catr", "circ", "vosp", "prof", "plan", "surf", "situ", "lum", "atm", "col"]
        
        df_for_prediction = df[model_features].copy()
        print(f"\n--- df_for_prediction avant transformations ---")
        print(f"Colonnes df_for_prediction: {df_for_prediction.columns.tolist()}")
        # print(df_for_prediction.info())
        # print(df_for_prediction.head())

        # Transformation binaire de 'atm' avant la normalisation
        conditions_advers_atm = [2.0, 3.0, 4.0, 5.0, 6.0, 9.0]
        if 'atm' in df_for_prediction.columns:
            print("Binarisation de 'atm'...")
            df_for_prediction['atm'] = df_for_prediction['atm'].apply(lambda x: 1.0 if x in conditions_advers_atm else 0.0)
            print(f"'atm' après binarisation (premières 5 lignes):\n{df_for_prediction['atm'].head().to_string()}")
            print(f"dtype de 'atm' après binarisation: {df_for_prediction['atm'].dtype}")
        
        if scaler is not None and columns_to_scale:
            print("\nNormalisation des données...")
            cols_to_normalize = [col for col in columns_to_scale if col in df_for_prediction.columns]
            if cols_to_normalize:
                print(f"Colonnes à normaliser: {cols_to_normalize}")
                df_for_prediction[cols_to_normalize] = scaler.transform(df_for_prediction[cols_to_normalize])
                print(f"Données normalisées pour les colonnes: {cols_to_normalize}")
            else:
                print("Aucune colonne à normaliser trouvée ou applicable.")
        # print(f"df_for_prediction après normalisation (premières 5 lignes):\n{df_for_prediction.head().to_string()}")
        
        # Prepare features (X) pour la prédiction
        cols_to_encode = ['catu', 'sexe', 'trajet']
        actual_cols_to_encode = [col for col in cols_to_encode if col in df_for_prediction.columns]
        print(f"\n--- Préparation pour get_dummies ---")
        print(f"Colonnes avant get_dummies: {df_for_prediction.columns.tolist()}")
        print(f"Colonnes à encoder (actual_cols_to_encode): {actual_cols_to_encode}")
        
        X_processed = pd.get_dummies(df_for_prediction, columns=actual_cols_to_encode, drop_first=True, dtype=float)
        print(f"\n--- X_processed après get_dummies ---")
        print(f"Colonnes X_processed: {X_processed.columns.tolist()}")
        # print(X_processed.info())
        # print(X_processed.head().to_string())

        # S'assurer que toutes les features attendues par le modèle sont présentes et dans le bon ordre
        try:
            model_expected_features = model.feature_names_in_.tolist()
            print(f"\n--- Alignement avec model.feature_names_in_ ---")
            print(f"model.feature_names_in_: {model_expected_features}")
        except AttributeError:
            print("Erreur: model.feature_names_in_ n'est pas disponible. Vérifier le modèle.")
            raise HTTPException(status_code=500, detail="Attribut feature_names_in_ manquant sur le modèle.")

        for col_name in model_expected_features:
            if col_name not in X_processed.columns:
                print(f"Ajout de la colonne manquante '{col_name}' à X_processed avec des zéros.")
                X_processed[col_name] = 0.0
        
        print(f"Colonnes X_processed après ajout des manquantes: {X_processed.columns.tolist()}")
        
        X_final = X_processed[model_expected_features]
        print(f"\n--- X_final avant prédiction ---")
        print(f"Colonnes X_final: {X_final.columns.tolist()}")
        # print(X_final.info())
        # print(X_final.head().to_string())

        # Faire les prédictions avec le modèle
        print("Appel de model.predict(X_final)...")
        y_pred_raw = model.predict(X_final)
        print("Prédictions brutes obtenues.")
        
        y_pred = y_pred_raw

        result_df = df.copy()
        result_df['prediction_grav'] = y_pred
        result_df['prediction_label'] = ['grave' if x == 1 else 'non_grave' for x in y_pred]
        
        result_df.to_csv("data/api_out/predictions.csv", index=False)
        print("Résultats enregistrés dans predictions.csv")
        print("--- Fin du traitement predict_csv ---")

        return {
            "user": current_user.username, 
            "message": "Prédiction effectuée avec succès",
            "predictions": y_pred.tolist(),
            "predictions_labels": ["grave" if x == 1 else "non_grave" for x in y_pred],
            "nb_predictions_grav_1": int(sum(y_pred == 1)),
            "nb_predictions_grav_0": int(sum(y_pred == 0))
        }

    except ImportError as e:
        print(f"ImportError: {str(e)}")
        return JSONResponse({'error': str(e)}, status_code=500)
    except ValidationError as e:
        print(f"ValidationError: {e.errors()}")
        raise HTTPException(status_code=422, detail=e.errors()) from e
    except Exception as e:
        import traceback
        print(f"Exception générale: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) from e

# Endpoint pour mettre à jour le modèle
@app.get("/protected/reload")
async def reload_model(current_user: User = Depends(get_current_user)):
    model = joblib.load(f"{model_dir}/best_model_2023.joblib")
    joblib.dump(model, "best_model_2023.joblib")

    return {"message": "Modèle rechargé avec succès"}

# Lancer l'application avec Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
