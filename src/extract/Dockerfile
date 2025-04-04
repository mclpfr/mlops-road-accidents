# Utiliser une image Python avec build-essential pré-installé
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier des dépendances depuis la racine du projet
COPY requirements.txt .

# Configurer OpenBLAS pour utiliser un seul thread
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

# Installer les dépendances Python avec versions spécifiques
RUN pip install --no-cache-dir --progress-bar off \
    beautifulsoup4==4.13.3 \
    pandas==1.5.3 \
    requests==2.31.0 \
    pyyaml==6.0.1 \
    numpy==1.21.6

# Copier le script spécifique à ce service
COPY src/extract/extract_data.py .

# Le point d'entrée sera défini dans docker-compose
# CMD ["python", "extract_data.py"]
