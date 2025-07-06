from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt


router = APIRouter()

# Configuration de la sécurité
JWT_SECRET_KEY = "key"
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Context de hachage des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Modèle Pydantic pour les données utilisateur
class User(BaseModel):
    username: str
    hashed_password: str
    role: str = "user"  # Par défaut, le rôle est 'user'

class Token(BaseModel):
    access_token: str
    token_type: str

# Exemple de base de données utilisateur
fake_users_db = {
    "user1": {
        "username": "user1",
        "hashed_password": pwd_context.hash("pass1"),
        "role": "user"
        },
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("adminpass"),
        "role": "admin"
        },
    "johndoe": {
        "username": "johndoe",
        "hashed_password": pwd_context.hash("johnsecret"),
        "role": "user"
        }
    }

# OAuth2 schema pour la récupération du token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

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
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

# Fonction pour obtenir l'utilisateur courant
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=username)
    if user is None:
        raise credentials_exception
    return user

@router.get("/")
def verify_api():
    return {"message": "Bienvenue ! L'API authentification est fonctionnelle."}

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # Vérifiez les informations d'identification de l'utilisateur ici
    if authenticate_user(fake_db=fake_users_db, username=form_data.username, password=form_data.password):
        user = get_user(fake_users_db, form_data.username)
        role = user.role if user else "user"
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
            )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username, "role": role},
        expires_delta=access_token_expires
        )
    return {"access_token": access_token, "token_type": "bearer"}
