from datetime import datetime, timezone, timedelta
from passlib.context import CryptContext
import requests
import jwt

# Clé secrète pour les tests JWT
JWT_SECRET_KEY = "key"
JWT_ALGORITHM = "HS256"

# URL pour les endpoints /login et /predict
LOGIN_URL = "http://127.0.0.1:7999/auth/token"
PREDICT_URL = "http://127.0.0.1:8000/protected/predict"

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

# Exemple
# fake_users_db = {
#     "user1": {
#         "username": "user1",
#         "hashed_password": pwd_context.hash("johnsecret")
#     }
# }

# def create_jwt_token(data: dict, expires_delta: datetime.timedelta = None):
#     '''Créer un token JWT'''
#     to_encode = data.copy()
#     if expires_delta:
#         expire = datetime.datetime.utcnow() + expires_delta
#     else:
#         expire = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
#     return encoded_jwt

# Fonction pour vérifier le mot de passe
# def verify_password(plain_password, hashed_password):
#     return pwd_context.verify(plain_password, hashed_password)

# Fonction pour obtenir un utilisateur
# def get_user(db, username: str):
#     if username in db:
#         user_dict = db[username]
#         return User(**user_dict)

# Fonction pour authentifier un utilisateur
# def authenticate_user(fake_db, username: str, password: str):
#     user = get_user(fake_db, username)
#     if not user or not verify_password(password, user.hashed_password):
#         return False
#     return user

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

def test_missing_jwt_token():
    '''Test de l'authentification JWT : jeton manquant'''
    response = requests.post(PREDICT_URL, json={}, timeout=10)
    assert response.status_code == 403

def test_invalid_jwt_token():
    '''Test de l'authentification JWT : jeton invalide'''
    headers = {"Authorization": "Bearer invalid_token"}
    response = requests.post(PREDICT_URL, headers=headers, json={}, timeout=10)
    assert response.status_code == 401

def test_expired_jwt_token():
    '''Test de l'authentification JWT : jeton expiré'''
    data = {"sub": "user1"}
    token = create_access_token(data, expires_delta=timedelta(seconds=-1))
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(PREDICT_URL, headers=headers, json={}, timeout=10)
    assert response.status_code == 401

def test_valid_jwt_token():
    '''Test de l'authentification JWT : jeton valide'''
    data = {"sub": "user1"}
    token = create_access_token(data)
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(PREDICT_URL, headers=headers, json={}, timeout=10)
    assert response.status_code != 401

def test_login_success():
    '''Test de l'API de connexion : identifiants corrects'''
    response = requests.post(
        LOGIN_URL,
        data=valid_credentials,
        timeout=10
    )
    assert response.status_code == 200

def test_login_failure():
    '''Test de l'API de connexion : identifiants incorrects'''
    response = requests.post(
        LOGIN_URL,
        data=invalid_credentials,
        timeout=10
    )
    assert response.status_code == 401
