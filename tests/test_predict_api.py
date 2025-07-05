from datetime import timedelta
from passlib.context import CryptContext
import requests


# URL pour le endpoint /predict
PREDICT_URL = "http://127.0.0.1:8000/protected/predict"

# Context de hachage des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Données de test pour l'authentification
valid_credentials = {
    "username": "johndoe",
    # "hashed_password": pwd_context.hash("johnsecret")
    "password": "johnsecret"
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

def test_missing_jwt_token():
    '''Test de l'authentification JWT : jeton manquant'''
    response = requests.post(PREDICT_URL, json={}, timeout=10)
    assert response.status_code == 401

def test_invalid_jwt_token():
    '''Test de l'authentification JWT : jeton invalide'''
    headers = {"Authorization": "Bearer invalid_token"}
    response = requests.post(PREDICT_URL, headers=headers, json={}, timeout=10)
    assert response.status_code == 401

def test_expired_jwt_token():
    '''Test de l'authentification JWT : jeton expiré'''
    data = {"sub": "johndoe"}
    token = create_access_token(data, expires_delta=timedelta(seconds=-1))
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(PREDICT_URL, headers=headers, json={}, timeout=10)
    assert response.status_code == 401

def test_valid_jwt_token():
    '''Test de l'authentification JWT : jeton valide'''
    data = {"sub": "johndoe"}
    token = create_access_token(data)
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(PREDICT_URL, headers=headers, json={}, timeout=10)
    assert response.status_code != 401

def test_predict_with_valid_data():
    '''Test de l'API de prédiction : données d'entrée correctes'''
    data = {"sub": "johndoe"}
    token = create_access_token(data)
    headers = {"Authorization": f"Bearer {token}"}
    test_data = valid_data
    response = requests.post(PREDICT_URL, headers=headers, json=test_data, timeout=10)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_with_invalid_data():
    '''Test de l'API de prédiction : données d'entrée invalides'''
    data = {"sub": "johndoe"}
    token = create_access_token(data)
    headers = {"Authorization": f"Bearer {token}"}
    test_data = invalid_data
    response = requests.post(PREDICT_URL, headers=headers, json=test_data, timeout=10)
    assert response.status_code == 422

def test_predict_server_error():
    '''Test de l'API : erreur serveur 5xx (predict_csv avec fichier vide)'''
    data = {"sub": "johndoe"}
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
    assert response.status_code == 401
