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

## Utilisation
### Extraction des Données
```bash
python src/extract_data.py
```

### Prétraitement des Données
```bash
python src/prepare_data.py
```

### Entraînement du Modèle
```bash
python src/train_model.py
```

### Lancement API
```bash
cd src
uvicorn api:app --reload
```

### Lancement de tests unitaires
```bash
cd tests
pytest tests.py
```

## Utilisation avec Docker Compose

Vous pouvez utiliser Docker Compose pour gérer les différents microservices du projet.

### Exécuter tous les services

Pour lancer tous les services définis dans le fichier `docker-compose.yml` (en respectant les dépendances) :

```bash
docker-compose up -d
```
Pour lancer un seul microservice *sans* démarrer ses dépendances :

```bash
docker-compose up --no-deps prepare_data
```

Pour arrêter les services :

```bash
docker-compose down
```
