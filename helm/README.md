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

## Désinstallation

```bash
helm uninstall mlops-road-accidents
```

## Notes

- Les volumes persistants ne sont pas automatiquement supprimés lors de la désinstallation
- Les secrets et configmaps doivent être gérés séparément
- Assurez-vous que les images Docker sont accessibles depuis votre cluster 