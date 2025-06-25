#!/bin/bash
set -e

# Créer les répertoires de logs avec les bonnes permissions
mkdir -p /opt/airflow/logs /opt/airflow/logs/scheduler /opt/airflow/logs/webserver /opt/airflow/logs/worker
chmod -R 777 /opt/airflow/logs

# Database initialization and user creation steps are now managed in /opt/airflow/scripts/entrypoint.sh

echo "Log directories preparation completed. No further action in init.sh."
exit 0
