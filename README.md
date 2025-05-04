# Projet de Prédiction des Accidents de la Route - MLOps

## Aperçu du Projet
Ce projet vise à prédire la gravité des accidents de la route en France en utilisant des techniques de machine learning. L'objectif est de fournir une estimation des urgences en temps réel pour les services de police et médicaux.

## Prérequis
- Python 3.10
- Bibliothèques Python : 
  - pandas
  - scikit-learn
  - numpy
  - requests
  - beautifulsoup4
  - pytest
  - fastapi
  - uvicorn
  - pyyaml
  - python-multipart

## Installation
1. Clonez le dépôt
```bash
git clone https://github.com/mclpfr/mlops-road-accidents.git
```
2. Créez un environnement virtuel
```bash
python3.10 -m venv venv
source venv/bin/activate
```
3. Installez les dépendances
```bash
pip install -r requirements.txt
```

## Configuration

1. Copiez le fichier exemple pour créer votre configuration
```bash
cp config.yaml.example config.yaml
```

2. Modifiez le fichier `config.yaml` :
   - Section `mlflow`: Remplacez `USERNAME`, `REPOSITORY` et `YOUR_DAGSHUB_TOKEN` par vos identifiants DagsHub
   - Section `postgresql`: Modifiez le mot de passe si nécessaire
   - Section `dagshub`: Remplacez `USERNAME` et `YOUR_DAGSHUB_TOKEN` par vos identifiants DagsHub

Le fichier `config.yaml` contient toutes les configurations nécessaires pour l'extraction des données, le suivi MLflow, la base de données PostgreSQL et l'intégration avec DagsHub.

## Utilisation
### Extraction des Données
```bash
python src/extract_data/extract_data.py
```

### Prétraitement des Données
```bash
python src/prepare_data/prepare_data.py
```

### Entraînement du Modèle
```bash
python src/train_model/train_model.py
```

### Lancement API
```bash
cd src/api
uvicorn api:app --reload
```

### Lancement de tests unitaires
```bash
cd tests
pytest tests.py
```

## Docker Compose

### Exécution complète du pipeline

Pour exécuter tout le pipeline de bout en bout :

```bash
docker-compose up --build
```

### Exécution d'un service spécifique

Pour lancer uniquement un service particulier :

```bash
docker-compose up --build <service>
```

Exemples :
```bash
# Extraction des données
docker-compose up --build extract_data

# Préparation des données
docker-compose up --build prepare_data

# Entraînement du modèle
docker-compose up --build train_model

# Versionnage DVC
docker-compose up --build auto_dvc
```
