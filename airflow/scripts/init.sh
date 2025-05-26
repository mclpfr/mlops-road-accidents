#!/bin/bash
set -e

# Créer les répertoires de logs avec les bonnes permissions
mkdir -p /opt/airflow/logs /opt/airflow/logs/scheduler /opt/airflow/logs/webserver /opt/airflow/logs/worker
chmod -R 777 /opt/airflow/logs

# Initialiser la base de données Airflow
airflow db init

# Créer un utilisateur administrateur
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com

echo "Initialisation d'Airflow terminée avec succès!"
exit 0
