#!/bin/bash
set -e

mkdir -p /opt/airflow/logs /opt/airflow/logs/scheduler /opt/airflow/logs/webserver /opt/airflow/logs/worker
chmod -R 777 /opt/airflow/logs

EVIDENTLY_DIR="/opt/project/evidently/current"
mkdir -p "$EVIDENTLY_DIR"
chown -R 50000:0 "$EVIDENTLY_DIR" || echo "Warn: chown failed, continuing"
chmod -R g+w "$EVIDENTLY_DIR"

# Database initialization and user creation steps are now managed in /opt/airflow/scripts/entrypoint.sh

echo "Log directories preparation completed. No further action in init.sh."
exit 0
