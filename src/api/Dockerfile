# Utiliser une image Python avec build-essential pré-installé
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier des dépendances depuis la racine du projet
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --progress-bar off \
    fastapi==0.68.1 \
    uvicorn==0.15.0 \
    pandas==1.5.3 \
    pyyaml==6.0.1 \
    scikit-learn==1.0.2 \
    numpy==1.21.6

# Copier le script spécifique à ce service
COPY src/api/api.py .

# Exposer le port que l'API utilisera
EXPOSE 8000

# Le point d'entrée sera défini dans docker-compose
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
