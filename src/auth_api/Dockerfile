# Utiliser une image de base officielle de Python
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier caché dans le conteneur
# COPY .config.json .

# Copier les fichiers de l'application dans le conteneur
COPY requirements.txt .

# Configuration des limites de mémoire et d'optimisation
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONMEM=256m

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/data /app/data/out

# Copier le fichier de configuration
COPY config.yaml .

# Copier le reste de l'application
COPY src/auth_api/main.py .
COPY src/auth_api/auth_api.py .

# Exposer le port sur lequel l'application va tourner
EXPOSE 7999

# Commande pour lancer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7999"]
