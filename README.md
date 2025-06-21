# MLOps - Prédiction des Accidents de la Route

## Aperçu du Projet

Ce projet MLOps complet prédit la gravité des accidents de la route en France en utilisant des données gouvernementales officielles. L'objectif est de fournir une estimation des urgences en temps réel pour optimiser les interventions des services de police et médicaux.

## Objectifs

- **Prédiction binaire** : Classifier les accidents comme "Grave" (hospitalisé/décédé) ou "Pas Grave" (indemne/blessé léger)
- **Pipeline automatisé** : Extraction, données synthétiques, préparation, entraînement et déploiement automatiques
- **Monitoring continu** : Détection de drift et re-entraînement automatique
- **API REST** : Service de prédiction sécurisé avec authentification JWT

## Architecture du Système

```
Extraction → Données Synthétiques → Préparation → Entraînement → API → Monitoring
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

3. **Lancement complet**
```bash
# Pipeline complet (extraction → entraînement → API → monitoring)
docker-compose up --build

# Ou en arrière-plan
docker-compose up -d --build
```

### Services Disponibles

| Service | URL | Description |
|---------|-----|-------------|
| API Prédictions | http://localhost:8000/docs | API REST avec documentation Swagger |
| Grafana | http://localhost:3000 | Dashboards de monitoring (admin/admin) |
| Prometheus | http://localhost:9090 | Métriques système et modèle |
| Airflow | http://localhost:8080 | Orchestration des pipelines (admin/admin) |

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
- Déduplication sur `Num_Acc`
- Imputation des valeurs manquantes par le mode
- Binarisation de la cible : `grav` → 0 (pas grave) / 1 (grave)
- Standardisation avec `StandardScaler`

## Modélisation

**Algorithme** : Random Forest Classifier
- **Hyperparamètres optimisés** : n_estimators=300, max_depth=15, etc.
- **Métriques** : Accuracy ~84%, Precision/Recall équilibrés
- **Validation** : Train/test split 80/20 avec stratification

**MLflow Integration** :
- Tracking automatique des expériences
- Model Registry avec versioning
- Tag "best_model" pour le meilleur modèle
- Transition automatique vers "Production"
- **DVC automatique** : Versioning des fichiers lors de la promotion d'un nouveau modèle

## Pipeline MLOps

Le projet utilise une approche innovante avec **génération de données synthétiques** pour simuler un environnement de production réaliste.

### Pourquoi des Données Synthétiques ?

**Problématique** : Les données de 2023 sont figées et statiques. Sans variation, le pipeline produirait toujours le même modèle avec la même accuracy, ce qui ne reflète pas la réalité d'un système en production.

**Solution** : Génération de 50% de données synthétiques qui permet de :
- **Varier les performances** : Accuracy différente à chaque entraînement
- **Simuler la dérive** : Évolution naturelle des distributions de données
- **Tester les alertes** : Déclenchement automatique du re-entraînement
- **Valider le monitoring** : Système de détection de drift opérationnel

### Étapes du Pipeline

1. **Extraction** (`extract_data`) : Téléchargement des données gouvernementales CSV 2023
2. **Augmentation** (`synthet_data`) : Génération de 50% données synthétiques basées sur les distributions réelles
3. **Préparation** (`prepare_data`) : Feature engineering et normalisation
4. **Entraînement** (`train_model`) : Random Forest avec tracking MLflow et variations d'accuracy
   - **DVC automatique** : Versioning des données et modèles lors de la promotion
   - **Git integration** : Commits automatiques avec métadonnées du modèle
5. **Import** (`import_data`) : Stockage PostgreSQL pour les dashboards
6. **Monitoring** : Détection de drift et alertes automatiques

### 1. Pipeline Principal (DAG `road_accidents`)
```
Extraction → Données Synthétiques → Préparation → Entraînement → Import DB → Monitoring
```

### 2. Pipeline Quotidien (DAG `daily_data_processing`)
```
Extraction Quotidienne → Données Synthétiques → Préparation → Mise à jour Evidently
```

### 3. Déclenchement Automatique
- **Seuil de drift** : > 0.5
- **Action** : Re-entraînement automatique via webhook Alertmanager → Airflow

## Monitoring & Alertes

### Détection de Drift avec Evidently

#### Architecture de la Détection de Drift

Le système utilise **Evidently** pour détecter automatiquement la dérive des données en comparant les distributions entre :
- **Données de référence** : Dataset utilisé pour entraîner le "meilleur modèle" 
- **Données courantes** : Nouvelles données traitées quotidiennement

#### Mise en Place Technique

**1. API Evidently** (`evidently/api/api.py`)
- Service FastAPI dédié exposé sur le port 8001
- Endpoint `/metrics` pour Prometheus au format OpenMetrics
- Calcul du score de drift basé sur les 13 features du modèle

**2. Stockage des Données**
```
evidently/
├── reference/          # Données de référence (best_model_data.csv)
├── current/           # Données courantes (current_data.csv)
└── api/              # Service de calcul du drift
```

**3. Calcul du Score de Drift**
- **Variables numériques** : Corrélation entre histogrammes (seuil < 0.95)
- **Variables catégorielles** : Distance de variation totale (seuil > 0.1)
- **Score final** : Proportion de features ayant dérivé

**4. Mise à Jour Automatique**
- **Données courantes** : Mises à jour par le DAG quotidien `daily_data_processing`
- **Données de référence** : Mises à jour lors de la promotion d'un nouveau meilleur modèle

#### Intégration avec le Pipeline MLOps

**1. Collecte par Prometheus**
```yaml
# monitoring/prometheus/prometheus.yml
scrape_configs:
  - job_name: 'evidently_drift_api'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['evidently-api:8001']
```

**2. Règles d'Alerte**
```yaml
# monitoring/prometheus/alert.rules.yml
- alert: HighDataDrift
  expr: ml_data_drift_score > 0.5
  for: 1m
  annotations:
    summary: "High data drift detected ({{ $value }})"
```

**3. Webhook Automatique**
```yaml
# monitoring/alertmanager/alertmanager.yml
receivers:
- name: 'airflow_webhook'
  webhook_configs:
  - url: 'http://evidently-api:8001/trigger_airflow_from_alert'
```

#### Workflow de Détection et Réaction

```
Nouvelles Données → Evidently API → Score Drift > 0.5 → Prometheus Alerte → 
Alertmanager → Webhook → Airflow DAG road_accidents → Re-entraînement → Nouveau Modèle
```

**Orchestration Airflow** :
- **DAG `daily_data_processing`** : Exécution quotidienne pour récupérer de nouveaux datasets (Extraction → Synthétique → Préparation → Evidently)
- **DAG `road_accidents`** : Déclenchement sur alerte de drift pour re-entraînement complet

#### Tests et Simulation

**Forcer un Drift Artificiel** (pour tests)
```bash
# Augmenter le drift à 80%
curl -X POST http://localhost:8001/config/noise \
  -H "Content-Type: application/json" \
  -d '{"noise": 0.8}'

# Vérifier l'alerte dans Prometheus
curl http://localhost:9090/api/v1/query?query=ml_data_drift_score
```

### Dashboards Grafana

#### Accès et Configuration
- **URL** : http://localhost:3000
- **Identifiants** : admin/admin
- **Sources de données** : PostgreSQL, Prometheus, Loki (pré-configurées)

#### Liste des Dashboards

**1. API Monitoring Dashboard**
- **Objectif** : Surveillance en temps réel de l'API de prédiction
- **Métriques clés** :
  - Uptime de l'API et temps de fonctionnement
  - Score de drift des données (gauge avec seuils d'alerte)
  - Taux de requêtes par seconde (RPS)
  - Latence P95 et temps de réponse moyen
  - Taux d'erreurs 5xx et distribution des codes HTTP
  - Utilisation CPU et mémoire des conteneurs API
  - **Logs API en temps réel** via Loki (panel logs intégré)

**2. Best Model Metrics Dashboard**
- **Objectif** : Suivi des performances du modèle en production
- **Métriques clés** :
  - Version et année du meilleur modèle
  - Accuracy, Precision, Recall, F1-Score
  - Évolution des métriques dans le temps
  - Seuils de performance avec alertes visuelles (vert/orange/rouge)

**3. Accidents 2023 Dashboard**
- **Objectif** : Analyse des données métier et statistiques descriptives
- **Métriques clés** :
  - Nombre total d'accidents analysés
  - Répartition par conditions de luminosité (pie chart)
  - Top 10 des départements les plus accidentés
  - Statistiques mises à jour toutes les 5 secondes

**4. Container Resources Dashboard**
- **Objectif** : Monitoring de l'infrastructure et des ressources système
- **Métriques clés** :
  - Utilisation CPU par conteneur (%)
  - Consommation mémoire par service
  - I/O réseau (réception/transmission)
  - I/O disque (lecture/écriture)
  - Vue d'ensemble de la santé de l'infrastructure

#### Alertes Visuelles
- **Seuils configurés** : Performance modèle (>80%), CPU (>70%), mémoire, drift (>0.3)
- **Codes couleur** : Vert (normal), Orange (attention), Rouge (critique)
- **Actualisation** : Toutes les 5 secondes pour un monitoring temps réel

### Métriques Système
```promql
# Score de drift des données
ml_data_drift_score

# Métriques API
http_requests_total
http_request_duration_seconds
```

### Alertes Grafana
- Drift élevé détecté
- Latence API anormale
- Erreurs 5xx en augmentation

## API REST

### Authentification
```bash
curl -X POST http://localhost:8000/auth/token \
  -d "username=johndoe&password=johnsecret"
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
pytest tests/test_api.py -v

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
│   ├── api/                 # API FastAPI
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

1. **DagsHub Setup** (Obligatoire)
   - Créer un compte sur [DagsHub](https://dagshub.com)
   - Créer un nouveau repository
   - Générer un token d'accès dans Settings > Access Tokens
   - Mettre à jour `mlflow.tracking_uri` avec ton URL MLflow

2. **GitHub Integration** (Optionnel)
   - Générer un Personal Access Token avec droits repo
   - Configurer `git.user.token` pour les commits automatiques DVC

3. **PostgreSQL** (Optionnel)
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

## Commandes Utiles

### Lancement Complet
```bash
# Lancement de tous les services (pipeline complet)
docker-compose up
```

### Services Individuels
```bash
# 1. Extraction des données
docker-compose up extract_data

# 2. Génération de données synthétiques
docker-compose up synthet_data

# 3. Préparation des données
docker-compose up prepare_data

# 4. Entraînement du modèle
docker-compose up train_model

# 5. Import en base de données
docker-compose up import_data

# API et base de données
docker-compose up api postgres

# Monitoring complet
docker-compose up prometheus grafana evidently-api

# Orchestration
docker-compose up airflow-webserver airflow-scheduler
```

### Gestion DVC (Automatique)
```bash
# DVC est géré automatiquement par train_model.py
# Lors de la promotion d'un nouveau meilleur modèle :
# - dvc commit --force pour les nouveaux fichiers
# - git add des fichiers .dvc
# - git commit avec métadonnées du modèle
# - git push automatique

# Récupération des données/modèles uniquement
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

## Troubleshooting

### Problèmes Courants

1. **Erreur de mémoire**
   ```bash
   # Augmenter la mémoire Docker à 8GB minimum
   ```

2. **MLflow non accessible**
   ```bash
   # Vérifier les credentials DagsHub dans config.yaml
   ```

3. **DVC non configuré**
   ```bash
   # DVC est géré automatiquement par train_model.py
   # Vérifier que git est configuré dans config.yaml
   ```

4. **Base de données non initialisée**
   ```bash
   docker-compose restart postgres
   ```

5. **Ports occupés**
   ```bash
   # Modifier les ports dans docker-compose.yml
   ```

## Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/amélioration`)
3. Commit (`git commit -m 'Ajout fonctionnalité'`)
4. Push (`git push origin feature/amélioration`)
5. Ouvrir une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Contact

**Marco LOPES** - MLOps & DevOps Engineer
- LinkedIn: [linkedin.com/in/marco-lopes-084063336](https://www.linkedin.com/in/marco-lopes-084063336/)

## Tags

`mlops` `machine-learning` `fastapi` `docker` `airflow` `mlflow` `prometheus` `grafana` `data-science` `road-safety` `france` `random-forest` `monitoring` `devops`
