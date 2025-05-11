import io
import json
import pandas as pd
import joblib
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
    # catu: int
    # sexe: int
    # trajet: int
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

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    hashed_password: str

# class UserCreate(BaseModel):
#     username: str
#     password: str

# Charger le modèle entraîné
model_dir = "models"
model = joblib.load(f"{model_dir}/best_model_2023.joblib")

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
        # Lire le fichier CSV avec pandas
        contents = await file_request.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        records = df.to_dict(orient='records')

        # Valider chaque enregistrement avec le modèle Pydantic
        features = [Feature(**record) for record in records]
        if not features:
            raise ValueError("None of the selected features are available in the dataset.")

        # Select model features
        model_features = [
            # "catu",
            # "sexe",
            # "trajet",
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
        df_ypred.to_csv("../../data/out/y_pred.csv", index=False)

        report = classification_report(y, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv("../../data/out/classification_report.csv")

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
@app.get("/protected/reload")
async def reload_model(current_user: User = Depends(get_current_user)):
    model = joblib.load(f"{model_dir}/best_model_2023.joblib")
    joblib.dump(model, "best_model_2023.joblib")

    return {"message": "Modèle rechargé avec succès"}

# Lancer l'application avec Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
