# MLOps Road Accidents Helm Chart

Ce chart Helm déploie l'infrastructure complète du projet MLOps Road Accidents sur un cluster Kubernetes.

## Prérequis

- Kubernetes 1.16+
- Helm 3.0+
- PV provisioner support dans le cluster
- Au moins 8Gi de stockage disponible

## Installation

```bash
# Ajouter le repo (si hébergé dans un registry Helm)
# helm repo add mlops-road-accidents <repository-url>

# Mettre à jour les dépendances
helm dependency update

# Installer le chart
helm install mlops-road-accidents ./helm
```

## Configuration

Les paramètres suivants peuvent être configurés dans le fichier `values.yaml` ou via la ligne de commande avec `--set`:

### Global

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| `global.userInfo.userId` | ID utilisateur pour les conteneurs | `"1000"` |
| `global.userInfo.groupId` | ID groupe pour les conteneurs | `"1000"` |

### Persistence

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| `persistence.postgres.enabled` | Activer le stockage persistant pour PostgreSQL | `true` |
| `persistence.postgres.size` | Taille du volume PostgreSQL | `1Gi` |
| `persistence.grafana.enabled` | Activer le stockage persistant pour Grafana | `true` |
| `persistence.grafana.size` | Taille du volume Grafana | `1Gi` |

### Services

Chaque service (extractData, synthetData, prepareData, trainModel, postgres, dataImport, grafana) peut être configuré avec:

- Image (repository et tag)
- Resources (requests et limits)
- Credentials (pour postgres et grafana)
- Ports de service

## Déploiement et Configuration d'Airflow

Cette section décrit les étapes pour déployer et configurer une instance Apache Airflow.

### 1. Prérequis

- Assurez-vous que Helm est installé et configuré.
- Assurez-vous que `kubectl` est configuré pour communiquer avec votre cluster Kubernetes.
- Le fichier de valeurs pour Airflow se trouve à `helm/airflow/values.yaml` (relatif à la racine du projet, ou `airflow/values.yaml` si vous êtes dans le répertoire `helm`). Vous pouvez l'adapter selon vos besoins (ex: ressources, configuration des sondes, etc.).

### 2. Installation du Chart Airflow

Le chart Airflow officiel (`apache-airflow/airflow`) est utilisé.

```bash
# Ajouter le dépôt Helm d'Apache Airflow (si ce n'est pas déjà fait)
helm repo add apache-airflow https://airflow.apache.org
helm repo update

# Installer le chart Airflow
# Exécuté depuis la racine du projet mlops-road-accidents:
helm install airflow apache-airflow/airflow \
  -n airflow \
  -f ./helm/airflow/values.yaml \
  --create-namespace

```

### 3. Accéder à l'Interface Utilisateur d'Airflow

Une fois les pods Airflow démarrés (en particulier `airflow-webserver`), vous pouvez accéder à l'interface utilisateur :

```bash
kubectl port-forward svc/airflow-webserver 8080:8080 -n airflow
```
Ouvrez ensuite [http://localhost:8080](http://localhost:8080) dans votre navigateur.
Les identifiants par défaut sont généralement `admin` / `admin`.

### 4. Copier un DAG dans Airflow (Exemple)

Pour ajouter un DAG à votre instance Airflow (par exemple, pour des tests rapides) :

1.  Identifiez le nom de votre pod scheduler Airflow :
    ```bash
    kubectl get pods -n airflow -l component=scheduler
    ```
    Notez le nom du pod (ex: `airflow-scheduler-xxxxxxxxxx-xxxxx`).

2.  Copiez votre fichier DAG (par exemple, `helm/airflow/dags/road_accidents.py` depuis la racine du projet) dans le pod scheduler :
    ```bash
    # Remplacez <NOM_DU_POD_SCHEDULER> par le nom réel du pod
    # Exécuté depuis la racine du projet mlops-road-accidents:
    kubectl cp ./helm/airflow/dags/road_accidents.py airflow/<NOM_DU_POD_SCHEDULER>:/opt/airflow/dags/road_accidents.py -c scheduler -n airflow
    
    # Si vous êtes dans le répertoire helm/, la commande pour copier le DAG serait:
    # kubectl cp ./airflow/dags/road_accidents.py airflow/<NOM_DU_POD_SCHEDULER>:/opt/airflow/dags/road_accidents.py -c scheduler -n airflow
    ```

**Important pour la production :** Pour une gestion des DAGs en production, il est recommandé d'utiliser la persistance des DAGs via un PersistentVolumeClaim ou la synchronisation avec un dépôt Git. Ces options sont configurables dans `helm/airflow/values.yaml` via les sections `dags.persistence` ou `dags.gitSync`. La méthode `kubectl cp` décrite ci-dessus est principalement destinée aux tests et développements rapides.

## Désinstallation

```bash
helm uninstall mlops-road-accidents
```

## Notes

- Les volumes persistants ne sont pas automatiquement supprimés lors de la désinstallation
- Les secrets et configmaps doivent être gérés séparément
- Assurez-vous que les images Docker sont accessibles depuis votre cluster 
