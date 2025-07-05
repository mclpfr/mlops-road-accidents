from datetime import datetime, timezone, timedelta
from passlib.context import CryptContext
import requests
import jwt

# Clé secrète pour les tests JWT
JWT_SECRET_KEY = "key"
JWT_ALGORITHM = "HS256"

# URL pour le endpoint /predict
PREDICT_URL = "http://127.0.0.1:8000/protected/predict"
RELOAD_URL = "http://127.0.0.1:8000/protected/reload"

# Context de hachage des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Données de test pour l'authentification
valid_credentials = {
    "username": "user1",
    # "hashed_password": pwd_context.hash("johnsecret")
    "password": "pass1"
    }

invalid_credentials = {
    "username": "wrong_user",
    # "hashed_password": pwd_context.hash("wrong_pwd")
    "password": "wrong_pwd"
    }

# Données de test pour la prédiction
valid_data = {
    "grav": 1,
    "catu": 1,
    "sexe": 1,
    "trajet": 1,
    "catr": 1,
    "circ": 1,
    "vosp": 1,
    "prof": 1,
    "plan": 1,
    "surf": 1,
    "situ": 1,
    "lum": 1,
    "atm": 1,
    "col": 1
}

invalid_data = {
    "grav": 1,
    "catu": 1,
    "sexe": 1,
    "trajet": 8,  # [-1, 0, 1, 2, 3, 4, 5, 9]
    "catr": 1,
    "circ": 1,
    "vosp": 1,
    "prof": 1,
    "plan": 1,
    "surf": 1,
    "situ": 1,
    "lum": 1,
    "atm": 1,
    "col": 1
}

# Fonction pour créer un token d'accès
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=1)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def test_predict_with_valid_data():
    '''Test de l'API de prédiction : données d'entrée correctes'''
    data = {"sub": "user1"}
    token = create_access_token(data)
    headers = {"Authorization": f"Bearer {token}"}
    test_data = valid_data
    response = requests.post(PREDICT_URL, headers=headers, json=test_data, timeout=10)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_with_invalid_data():
    '''Test de l'API de prédiction : données d'entrée invalides'''
    data = {"sub": "user1"}
    token = create_access_token(data)
    headers = {"Authorization": f"Bearer {token}"}
    test_data = invalid_data
    response = requests.post(PREDICT_URL, headers=headers, json=test_data, timeout=10)
    assert response.status_code == 422

def test_predict_server_error():
    '''Test de l'API : erreur serveur 5xx (predict_csv avec fichier vide)'''
    data = {"sub": "user1"}
    token = create_access_token(data)
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file_request": ("empty.csv", "", "text/csv")}
    response = requests.post(
        PREDICT_URL + "_csv",  # endpoint /protected/predict_csv
        headers=headers,
        files=files,
        timeout=10,
    )
    assert response.status_code >= 500 and response.status_code < 600

def test_predict_missing_jwt():
    '''Test de l'API de prédiction : jeton JWT manquant'''
    test_data = valid_data
    response = requests.post(PREDICT_URL, json=test_data, timeout=10)
    assert response.status_code == 403

def test_predict_reload_error():
    '''Test de l'API de rechargement du modèle : échec (utilisateur non autorisé)'''
    data = {"sub": "user1", "role": "user"}
    token = create_access_token(data)
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(RELOAD_URL, headers=headers, timeout=10)
    assert response.status_code == 403

def test_predict_reload_success():
    '''Test de l'API de rechargement du modèle : succès (avec rôle admin)'''
    data = {"sub": "user1", "role": "admin"}
    token = create_access_token(data)
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(RELOAD_URL, headers=headers, timeout=10)
    assert response.status_code == 200
