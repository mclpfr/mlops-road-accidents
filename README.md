# MLOps - Prédiction des Accidents de la Route

## Aperçu du Projet

Ce projet MLOps complet prédit la gravité des accidents de la route en France en utilisant des données gouvernementales officielles. L'objectif est de fournir une estimation des urgences en temps réel pour optimiser les interventions des services de police et médicaux.

## Objectifs

- **Prédiction binaire** : Classifier les accidents comme "Grave" (hospitalisé/décédé) ou "Pas Grave" (indemne/blessé léger)
- **Pipeline automatisé** : Extraction, données synthétiques, préparation, entraînement et déploiement automatiques
- **Monitoring complet** : 
  - Détection de drift des données avec Evidently
  - Surveillance des performances APIs en temps réel
  - Dashboards de monitoring système et métier
  - Re-entraînement automatique sur alerte de drift
- **API REST** :
  - Service d'authentification JWT
  - Service de prédiction sécurisé avec authentification JWT

## Architecture du Système

![Architecture du Système](images/archi_system.jpg)

### Architecture Technique Détaillée

```mermaid
graph TB
    %% Couche Data Sources
    subgraph "Data Sources"
        GOV["`**Données Gouvernementales**
        data.gouv.fr
        Accidents 2023`"]
    end

    %% Couche Pipeline ML
    subgraph "Pipeline ML (Airflow)"
        EXTRACT["`**extract_data**
        Téléchargement
        Fusion 4 CSV`"]
        SYNTHETIC["`**synthet_data**
        Génération 50/50
        Données synthétiques`"]
        PREPARE["`**prepare_data**
        Feature engineering
        Standardisation`"]
        TRAIN["`**train_model**
        Random Forest
        MLflow tracking`"]
        IMPORT["`**import_data**
        PostgreSQL
        Dashboards`"]
    end

    %% Couche MLOps
    subgraph "DagsHub"
        MLFLOW["`**MLflow**
        Model Registry
        Versioning
        Tracking`"]
        DVC["`**DVC**
        Data versioning
        Git intégration
        Reproductibilité`"]
    end

    %% Couche Storage
    subgraph "Storage Layer"
        POSTGRES["`**PostgreSQL**
        Données business
        Métriques modèle`"]
        FILES["`**File System**
        Modèles .joblib
        Données CSV`"]
        EVID_DATA["`**Evidently Data**
        Reference/Current
        Drift detection`"]
    end

    %% Couche API
    subgraph "API Services"
        AUTH["`**Auth API**
        Port 7999
        JWT tokens`"]
        PREDICT["`**Predict API**
        Port 8000
        Prédictions sécurisées`"]
    end

    %% Couche Monitoring
    subgraph "Monitoring Stack"
        PROMETHEUS["`**Prometheus**
        Port 9090
        Métriques collecte`"]
        GRAFANA["`**Grafana**
        Port 3000
        5 Dashboards`"]
        ALERT["`**Alertmanager**
        Webhook alerts
        Re-entraînement`"]
        LOKI["`**Loki**
        Log aggregation
        API logs`"]
        EVID_API["`**Evidently API**
        Port 8001
        Détection drift`"]
    end

    %% Couche Orchestration
    subgraph "Orchestration"
        AIRFLOW["`**Airflow**
        Port 8080
        DAGs pipeline`"]
    end

    %% Couche Users
    subgraph "Users & Interfaces"
        USERS["`**Users**`"]
        ADMIN["`**Admin**
        Monitoring dashboards
        MLOps oversight`"]
    end

    %% Flux de données
    GOV --> EXTRACT
    EXTRACT --> SYNTHETIC
    SYNTHETIC --> PREPARE
    PREPARE --> TRAIN
    TRAIN --> IMPORT
    TRAIN --> MLFLOW
    TRAIN --> DVC
    IMPORT --> POSTGRES
    TRAIN --> FILES
    PREPARE --> EVID_DATA
    USERS --> PREDICT
    PREDICT --> AUTH
    ADMIN --> GRAFANA
    GRAFANA --> PROMETHEUS
    GRAFANA --> LOKI
    PROMETHEUS --> PREDICT
    PROMETHEUS --> AUTH
    LOKI --> PREDICT
    LOKI --> AUTH
    ALERT --> PROMETHEUS
    EVID_API --> EVID_DATA
    AIRFLOW --> EXTRACT
    AIRFLOW --> SYNTHETIC
    AIRFLOW --> PREPARE
    AIRFLOW --> TRAIN
    AIRFLOW --> IMPORT
```


## Démarrage Rapide

### Prérequis

- Docker & Docker Compose
- Git
- 8 Go RAM minimum recommandé

### Installation

1. **Cloner le projet**
```bash
git clone https://github.com/mclpfr/mlops-road-accidents.git
cd mlops-road-accidents
```

2. **Configuration**
```bash
cp config.yaml.example config.yaml
```

Éditer le fichier `config.yaml` et remplacer :
- `USERNAME` : Nom d'utilisateur DagsHub
- `REPOSITORY` : Nom du repository MLflow sur DagsHub  
- `YOUR_DAGSHUB_TOKEN` : Token d'accès DagsHub
- `YOUR_POSTGRES_PASSWORD` : Mot de passe PostgreSQL (optionnel)
- `NAME` et `EMAIL` : Identifiants Git
- `YOUR_TOKEN` : Token GitHub pour les commits automatiques

3. **Lancement avec `Makefile`**

Le `Makefile` à la racine du projet simplifie la gestion des conteneurs. Voici les commandes principales :

#### Lancement Complet
Pour démarrer tous les services en arrière-plan :
```bash
make start-all
```

#### Gestion par Groupe de Services
Vous pouvez également démarrer, arrêter ou redémarrer des groupes de services spécifiques :

```bash
# Démarrer les services de monitoring (Prometheus, Grafana, etc.)
make start-monitoring

# Démarrer le pipeline ML (de l'extraction à l'entraînement)
make start-ml

# Démarrer l'interface utilisateur (Streamlit, API)
make start-ui

# Démarrer Airflow
make start-airflow
```

#### Arrêt et Nettoyage
```bash
# Arrêter un groupe de services (ex: monitoring)
make stop-monitoring

# Arrêter tous les services
make stop-all

# Arrêter et supprimer les conteneurs
make clean

# Arrêter, supprimer les conteneurs et les volumes de données
make purge
```
### Services Disponibles

| Service | URL | Description |
|---------|-----|-------------|
| API Authentification | http://localhost:7999/docs | API REST avec documentation Swagger |
| API Prédictions | http://localhost:8000/docs | API REST avec documentation Swagger |
| Grafana | http://localhost:3000 | 5 Dashboards de monitoring |
| Prometheus | http://localhost:9090 | Métriques système et modèle |
| Airflow | http://localhost:8080 | Orchestration des pipelines  |
| Evidently API | http://localhost:8001 | Service de détection de drift |

## Monitoring & Dashboards Grafana

### Accès et Configuration
- **URL** : http://localhost:3000
- **Sources de données** : PostgreSQL, Prometheus, Loki (pré-configurées)
- **Actualisation** : Toutes les 5 secondes pour un monitoring temps réel

### 5 Dashboards Intégrés

#### **1. API Performance Monitoring Dashboard**
**Surveillance en temps réel de l'API de prédiction**

**Métriques de Performance :**
- Uptime de l'API et temps de fonctionnement total
- Taux de requêtes par seconde (RPS)
- Latence P95 et temps de réponse moyen
- Taux d'erreurs 5xx et distribution des codes HTTP
- Utilisation CPU et mémoire des conteneurs API

**Logs en Temps Réel :**
- Panel logs API intégré via Loki
- Corrélation logs/métriques pour diagnostic rapide

**Alertes Visuelles :**
- Seuils CPU (>70%), mémoire (>80%), latence (>500ms)
- Codes couleur : Vert (normal), Orange (attention), Rouge (critique)

#### **2. Data Drift Monitoring Dashboard**
**Détection automatique de la dérive des données**

**Score de Drift Global :**
- Gauge principal avec score Evidently (0-1)
- Seuil d'alerte : >0.5 (déclenche re-entraînement automatique)
- Évolution du drift dans le temps (graphique temporel)

**Drift par Feature :**
- Tableau détaillé des 13 variables du modèle
- Status individuel : OK / DRIFT pour chaque feature
- Histogrammes de comparaison référence vs courant

**Pipeline de Détection :**
- Mise à jour quotidienne via DAG `daily_data_processing`
- Webhook automatique Alertmanager → Airflow sur drift élevé
- Logs de détection et actions automatiques

#### **3. Model Performance Dashboard**
**Suivi des performances ML en production**

**Métriques du Meilleur Modèle :**
- Version et timestamp du modèle en production
- Accuracy, Precision, Recall, F1-Score actuels

**Gestion des Versions :**
- Historique des promotions de modèles
- Déclencheurs de re-entraînement (drift/performance)
- Métadonnées d'entraînement (hyperparamètres, dataset size)

**Seuils de Performance :**
- Accuracy >85% (excellent)
- Accuracy 80-85% (acceptable)
- Accuracy <80% (re-entraînement recommandé)

#### **4. Business Analytics Dashboard**
**Analyse des données métier et statistiques descriptives**

**Vue d'Ensemble des Accidents :**
- Nombre total d'accidents analysés (2023)
- Répartition Grave vs Pas Grave (pie chart)
- Tendances temporelles et saisonnières

**Analyses Géographiques :**
- Top 10 des départements les plus accidentés
- Répartition par type de route (autoroute/nationale/etc.)
- Cartographie des zones à risque

**Conditions d'Accidents :**
- Répartition par conditions de luminosité
- Impact météorologique sur la gravité
- Analyse par catégorie d'usager (conducteur/piéton/passager)

### Intégration Complète du Monitoring


**Flux de Détection de Drift :**
1. **Collecte des Données** : 
   - Les nouvelles données sont collectées et stockées dans `evidently/current/current_data.csv`
   - Les données de référence (modèle de production) sont stockées dans `evidently/reference/best_model_data.csv`

2. **Flux de Détection de Drift :**
   - Ce processus est la première étape du **Drift Controller**, un workflow automatisé qui orchestre la détection, l'alerte et le ré-entraînement.
   - Le service Evidently API compare les distributions des caractéristiques entre les données de référence et courantes
   - Pour les variables numériques : Utilisation de la distance de Wasserstein (seuil > 0.1)
   - Pour les variables catégorielles : Test du Khi-deux (seuil p-value < 0.05)
   - Un score global de drift est calculé (0-1), où >0.5 indique un drift significatif

**API Evidently** (`http://localhost:8001`)
- Service FastAPI dédié au calcul de drift
- Endpoints principaux :
  - `POST /calculate_drift` : Calcule le drift entre les jeux de données
  - `GET /metrics` : Retourne les métriques au format Prometheus
  - `GET /drift_score` : Retourne le score de drift actuel
- Format des réponses en JSON pour intégration avec d'autres services

**Stockage des Données :**
```
evidently/
├── reference/    # Données de référence (best_model_data.csv)
├── current/     # Données courantes (current_data.csv)
└── api/         # Code source du service de calcul
```

**Métriques Surveillées :**
- `data_drift_score` : Score global de drift (0-1)
- `feature_drift{feature="NOM_DE_LA_FEATURE"}` : Score par caractéristique
- `drift_detection_timestamp` : Dernier calcul de drift

**Seuils d'Alerte :**
- **Avertissement** : Score > 0.3
- **Critique** : Score > 0.5 (déclenche le re-entraînement)

**Méthodes de Détection Avancées :**
- **Analyse des distributions** : Comparaison des distributions de probabilité
- **Test de Kolmogorov-Smirnov** : Pour détecter les changements dans les distributions
- **Distance de Jensen-Shannon** : Pour mesurer la similarité entre distributions
- **Analyse des corrélations** : Détection des changements dans les relations entre variables
- **Variables catégorielles** : Distance de variation totale (seuil > 0.1)
- **Score final** : Proportion de features ayant dérivé

#### **Tests de Drift (Simulation)**

```bash
# Forcer un drift artificiel pour tester les alertes
curl -X POST http://localhost:8001/config/noise \
  -H "Content-Type: application/json" \
  -d '{"noise": 0.8}'

# Vérifier l'alerte dans Grafana Dashboard "Data Drift Monitoring"
# L'alerte se déclenche automatiquement et lance le re-entraînement

# Réinitialiser le drift
curl -X POST http://localhost:8001/config/noise \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Métriques Prometheus Disponibles

```promql
# Drift des données
data_drift_score                    # Score global de drift (0-1)
feature_drift{feature="catu"}       # Drift par feature

# Performance API
http_requests_total                    # Nombre total de requêtes
http_request_duration_seconds          # Durée des requêtes
api_predictions_total                  # Nombre de prédictions

# Système
container_cpu_usage_seconds_total      # CPU par conteneur
container_memory_usage_bytes           # Mémoire par conteneur
```

### Workflow de Monitoring Automatique

1. **Monitoring Continu** : Les 4 dashboards actualisent leurs métriques toutes les 5 secondes
2. **Préparation des Données** : Airflow copie quotidiennement les données traitées dans le dossier `evidently/current/`
3. **Alertes Automatiques** : Prometheus surveille les seuils et déclenche Alertmanager
4. **Actions Correctives** : Webhook automatique vers Airflow pour re-entraînement
5. **Validation** : Nouveau modèle validé et promu automatiquement si performances meilleures

## Données

**Source** : [Données gouvernementales françaises](https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/)

**Variables utilisées** (13 features) :
- `catu` : Catégorie usager (conducteur/passager/piéton)
- `sexe` : Sexe (masculin/féminin)
- `trajet` : Motif du déplacement (domicile-travail/courses/loisirs/etc.)
- `catr` : Type de route (autoroute/nationale/départementale/communale)
- `circ` : Régime de circulation (sens unique/bidirectionnelle/chaussées séparées)
- `vosp` : Voie spécialisée (piste cyclable/voie bus/aucune)
- `prof` : Profil de la route (plat/pente/sommet de côte)
- `plan` : Tracé en plan (rectiligne/courbe à gauche/courbe à droite/en S)
- `surf` : État de la surface (normale/mouillée/flaques/enneigée/verglacée)
- `situ` : Situation de l'accident (sur chaussée/sur accotement/sur refuge)
- `lum` : Conditions d'éclairage (plein jour/crépuscule/nuit avec/sans éclairage)
- `atm` : Conditions météorologiques (normale/pluie/brouillard/neige)
- `col` : Type de collision (frontale/par l'arrière/par le côté/en chaîne)

**Préparation** :
- **Génération de données synthétiques** : Mélange 50% données réelles + 50% synthétiques
  - Permet de simuler la variabilité des données en production
  - Évite d'avoir toujours la même accuracy avec des données 2023 figées
  - Essentiel pour tester la détection de drift et les alertes automatiques
- Imputation des valeurs manquantes par le mode
- Binarisation de la cible : `grav` → 0 (pas grave) / 1 (grave)
- Standardisation avec `StandardScaler`

## Modélisation

**Algorithme** : Random Forest Classifier
- **Hyperparamètres optimisés** : n_estimators=300, max_depth=15, min_samples_split=5, min_samples_leaf=2, max_features="sqrt", random_state=42
- **Métriques** : Precision/Recall équilibrés avec performance variable selon l'entraînement
- **Validation** : Train/test split 80/20 avec stratification

**MLflow Integration** :
- Tracking automatique des expériences
- Model Registry avec versioning
- Tag "best_model" pour le meilleur modèle
- Transition automatique vers "Production"
- **DVC automatique** : Versioning des fichiers lors de la promotion d'un nouveau modèle

## Pipeline MLOps

Le projet utilise une approche avec **génération de données synthétiques** pour simuler un environnement de production réaliste.

### Pourquoi des Données Synthétiques ?

**Problématique** : Les données de 2023 sont figées et statiques. Sans variation, le pipeline produirait toujours le même modèle avec la même accuracy, ce qui ne reflète pas la réalité d'un système en production.

**Solution** : Génération de 50% de données synthétiques qui permet de :
- **Varier les performances** : Accuracy différente à chaque entraînement
- **Simuler la dérive** : Évolution naturelle des distributions de données
- **Tester les alertes** : Déclenchement automatique du re-entraînement
- **Valider le monitoring** : Système de détection de drift opérationnel

### Étapes du Pipeline

1. **Extraction** (`extract_data`) : Téléchargement et fusion des 4 fichiers CSV gouvernementaux 2023 (caractéristiques, lieux, usagers, véhicules) en un seul fichier unifié `accidents_2023.csv`
2. **Augmentation** (`synthet_data`) : Génération de 50% données synthétiques basées sur les distributions réelles
3. **Préparation** (`prepare_data`) : Feature engineering et normalisation
4. **Entraînement** (`train_model`) : Random Forest avec tracking MLflow et variations d'accuracy
   - **DVC automatique** : Versioning des données et modèles lors de la promotion
   - **Git integration** : Commits automatiques avec métadonnées du modèle
5. **Import** (`import_data`) : Stockage PostgreSQL pour les dashboards
6. **Monitoring** : Détection de drift et alertes automatiques

### 1. Pipeline Principal (DAG `road_accidents`)

![Schéma du DAG Airflow](images/dags_road.jpg)

### 2. Pipeline Quotidien (DAG `daily_data_processing`)

![Schéma du DAG Quotidien Airflow](images/dags_daily.jpg)

### 3. Déclenchement Automatique
- **Seuil de drift** : > 0.5
- **Action** : Re-entraînement automatique via webhook Alertmanager → Airflow

## API REST

### Authentification
```bash
curl -X POST http://localhost:7999/auth/token \
  -d "username=user1&password=pass1"
```

### Prédiction Unitaire
```bash
curl -X POST http://localhost:8000/protected/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "catu": 1, "sexe": 1, "trajet": 1,
    "catr": 3, "lum": 5, "atm": 2,
    "surf": 2, "col": 6, "circ": 2,
    "plan": 1, "vosp": 0, "prof": 1,
    "situ": 1
  }'
```

### Prédiction Batch (CSV)
```bash
curl -X POST http://localhost:8000/protected/predict_csv \
  -H "Authorization: Bearer $TOKEN" \
  -F "file_request=@accidents.csv"
```

## Tests

```bash
# Tests unitaires
pytest tests/ -v

# Test de l'API (après démarrage)
pytest tests/test_auth_api.py -v
pytest tests/test_predict_api.py -v

# Vérification des données
pytest tests/test_check_data_*.py -v
```

### Test de la Détection de Drift

```bash
# Forcer un drift artificiel pour tester les alertes
curl -X POST http://localhost:8001/config/noise \
  -H "Content-Type: application/json" \
  -d '{"noise": 0.8}'

# Réinitialiser le drift
curl -X POST http://localhost:8001/config/noise \
  -H "Content-Type: application/json" \
  -d '{}'
```


## Structure du Projet

```
mlops-road-accidents/
├── src/                     # Code source
│   ├── extract_data/        # Extraction des données gouvernementales
│   ├── synthet_data/        # Génération de données synthétiques (50/50)
│   ├── prepare_data/        # Préparation et feature engineering
│   ├── train_model/         # Entraînement et MLflow
│   ├── auth_api/            # API FastAPI d'authentification
│   ├── predict_api/         # API FastAPI de prédictions
│   ├── import_data/         # Import en base PostgreSQL
│   └── postgresql/          # Schéma base de données
├── airflow/                 # DAGs et configuration Airflow
├── monitoring/              # Configuration Prometheus/Grafana
├── evidently/               # API de détection de drift
├── tests/                   # Tests automatisés
├── docker-compose.yml       # Orchestration des services
├── dvc.yaml                 # Pipeline DVC
└── config.yaml             # Configuration générale
```

## Configuration Détaillée

### Étapes de Configuration

1. **DagsHub Setup** 
   - Créer un compte sur [DagsHub](https://dagshub.com)
   - Créer un nouveau repository
   - Générer un token d'accès dans Settings > Access Tokens
   - Mettre à jour `mlflow.tracking_uri` avec ton URL MLflow

2. **GitHub Integration** 
   - Générer un Personal Access Token avec droits repo
   - Configurer `git.user.token` pour les commits automatiques DVC

3. **PostgreSQL** 
   - Modifier `postgresql.password` si nécessaire
   - Utiliser les paramètres par défaut pour un démarrage rapide

### Variables Importantes (`config.yaml`)

```yaml
data_extraction:
  year: "2023"
  url: "https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/"

mlflow:
  tracking_uri: "https://dagshub.com/USERNAME/REPOSITORY.mlflow"
  username: "USERNAME"
  password: "YOUR_DAGSHUB_TOKEN"

model:
  type: "RandomForestClassifier"
  random_state: 42
  test_size: 0.2
  cv_folds: 5
  hyperparameters:
    n_estimators: 300
    max_depth: 15
    min_samples_split: 5
    min_samples_leaf: 2
    max_features: "sqrt"

postgresql:
  host: "postgres"
  port: "5432"
  user: "postgres"
  password: "YOUR_POSTGRES_PASSWORD"
  database: "road_accidents"

dagshub:
  user: "USERNAME"
  token: "YOUR_DAGSHUB_TOKEN"

git:
  user:
    name: "NAME"
    email: "EMAIL"
    token: "YOUR_TOKEN"
```

## Intégration Continue et Déploiement (CI/CD)

Le projet utilise GitHub Actions pour automatiser le processus d'intégration continue et de déploiement des images Docker. À chaque push sur la branche `main` ou lors d'une action manuelle, le workflow suivant est exécuté :

### Pipeline CI/CD

1. **Tests et Validation**
   - Exécution des tests automatisés avec `pytest`
   - Vérification de la qualité du code
   - Validation des configurations

2. **Construction des Images Docker**
   - Construction des images pour chaque service :
     - `extract_data`
     - `synthet_data`
     - `prepare_data`
     - `train_model`
     - `auth_api`
     - `predict_api`
     - `postgres`
     - `import_data`

3. **Versionnement et Publication**
   - Chaque image est taguée avec :
     - Le numéro du run GitHub Actions (`${{ github.run_number }}`)
     - Le tag `latest` pour la dernière version stable
   - Les images sont automatiquement poussées vers Docker Hub

### Configuration Requise

Pour que le pipeline fonctionne, les secrets suivants doivent être configurés dans les paramètres du dépôt GitHub :

- `DOCKER_USERNAME` : Votre nom d'utilisateur Docker Hub
- `DOCKER_TOKEN` : Votre token d'accès Docker Hub
- `CONFIG_YAML` : Contenu du fichier de configuration

### Déclenchement du Pipeline

Le pipeline se déclenche automatiquement dans les cas suivants :
- Push sur la branche `main`
- Pull request vers la branche `main`
- Déclenchement manuel via l'interface GitHub Actions

## Déploiement sur Kubernetes avec Helm

Le projet peut être déployé sur un cluster Kubernetes à l'aide du Chart Helm fourni. Cette méthode permet un déploiement standardisé et reproductible dans différents environnements.

### Prérequis

- Un cluster Kubernetes fonctionnel (v1.19+)
- Helm v3 installé
- `kubectl` configuré pour communiquer avec votre cluster

### Structure du Chart Helm

```
helm/
├── Chart.yaml             # Métadonnées du chart
├── values.yaml            # Valeurs par défaut
├── values-prod.yaml       # Valeurs pour l'environnement de production
├── templates/             # Templates Kubernetes
│   ├── _helpers.tpl       # Fonctions d'aide
│   ├── configmap.yaml     # ConfigMap pour les configurations
│   ├── deployment.yaml    # Déploiements des services
│   ├── ingress.yaml       # Configuration Ingress
│   ├── secret.yaml        # Secrets pour les credentials
│   ├── service.yaml       # Services Kubernetes
│   └── pvc.yaml           # Persistent Volume Claims
└── charts/                # Sous-charts (dépendances)
```

### Installation

```bash
# Vérifier la syntaxe du chart
helm lint ./helm

# Visualiser les manifestes générés sans installer
helm template mlops-accidents ./helm --values ./helm/values.yaml

# Installer le chart en environnement de développement
helm install mlops-accidents ./helm --values ./helm/values.yaml

# Installer en production avec des valeurs spécifiques
helm install mlops-accidents ./helm --values ./helm/values-prod.yaml
```

### Configuration

Le fichier `values.yaml` permet de personnaliser le déploiement :

```yaml
global:
  environment: dev
  storageClass: standard

image:
  registry: docker.io
  repository: mclpfr
  tag: latest
  pullPolicy: Always

replicas:
  predict_api: 2
  auth_api: 1
  evidently_api: 1

resources:
  predict_api:
    limits:
      cpu: 500m
      memory: 512Mi
    requests:
      cpu: 200m
      memory: 256Mi

postgres:
  enabled: true
  persistence:
    size: 10Gi

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
    dashboards:
      autoImport: true
```

### Mise à jour du déploiement

```bash
# Mettre à jour après modification des valeurs ou du code
helm upgrade mlops-accidents ./helm --values ./helm/values.yaml

# Rollback en cas de problème
helm rollback mlops-accidents 1
```

### Scaling

Le déploiement peut être facilement mis à l'échelle :

```bash
# Scaling horizontal de l'API de prédiction
kubectl scale deployment mlops-accidents-predict-api --replicas=5

# Ou via une mise à jour Helm
helm upgrade mlops-accidents ./helm --set replicas.predict_api=5
```

### Intégration avec ArgoCD (GitOps)

Pour une approche GitOps, le chart Helm peut être intégré avec ArgoCD :

```yaml
# Application ArgoCD (application.yaml)
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: mlops-accidents
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/mclpfr/mlops-road-accidents.git
    targetRevision: HEAD
    path: helm
  destination:
    server: https://kubernetes.default.svc
    namespace: mlops-accidents
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

## Commandes Utiles

### Gestion DVC (Automatique)

DVC est géré automatiquement par train_model.py lors de la promotion d'un nouveau meilleur modèle :
- dvc commit --force pour les nouveaux fichiers
- git add des fichiers .dvc
- git commit avec métadonnées du modèle
- git push automatique

Pour récupérer les données/modèles uniquement :
```bash
dvc pull
```

### Debugging
```bash
# Logs en temps réel
docker-compose logs -f [service_name]

# Accès aux conteneurs
docker-compose exec [service_name] bash

# Vérifier le versioning DVC (géré automatiquement)
git log --oneline | head -10
```

## Métriques de Performance

Le modèle Random Forest génère des métriques de performance qui **varient à chaque entraînement** grâce à la génération de données synthétiques :

| Métrique | Description |
|----------|-------------|
| **Accuracy** | Précision globale du modèle (variable selon l'entraînement) |
| **Precision** | Précision macro moyenne entre les classes |
| **Recall** | Rappel macro moyenne entre les classes |
| **F1-Score** | Score F1 macro moyenne |

**Avantage de la variabilité** : Cette variation permet de tester efficacement le système de détection de drift et la promotion automatique de modèles basée sur l'amélioration des performances.

## Limites et Axes d'Amélioration

- Les données synthétiques ne remplacent pas des données réelles de production
- Le modèle ne prend pas en compte les données en temps réel (ex : trafic, événements locaux)
- L'évaluation ne couvre pas les biais potentiels liés au genre ou à la région géographique
- Vérification automatique du schéma et des valeurs aberrantes dans prepare_data
- Tests de qualité des données via tests/test_check_data_*.py

## Sécurité

- L'API REST est protégée par JWT (JSON Web Token)
- Les données utilisateurs ne sont pas conservées
- L'accès aux services internes (Prometheus, Grafana, Airflow) est limité au réseau local
- Les secrets sont injectés via config.yaml et ne sont pas commités

## Reproductibilité

Pour reproduire l'entraînement avec les données et le code exacts d'un run antérieur :

```bash
git checkout <commit_hash>
dvc pull
docker-compose up train_model
```

## Ressources

- Environ 103 Mo de données après extraction
- Entraînement complet en ~3 minutes sur CPU
- Taille moyenne du modèle : 124 Mo
- RAM recommandée : 8 Go minimum

## Contact

**Marco LOPES** - MLOps & DevOps Engineer
- LinkedIn: [linkedin.com/in/marco-lopes-084063336](https://www.linkedin.com/in/marco-lopes-084063336/)

**Francis CHIN** - MLOps & Data Scientist
