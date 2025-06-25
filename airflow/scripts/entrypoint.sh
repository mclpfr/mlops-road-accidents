#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if current user is root
if [ "$(id -u)" = "0" ]; then
    # Configure docker group for socket access (as root)
    echo "Configuring Docker socket access..."
    if [ -S /var/run/docker.sock ]; then
        # Get the GID of the group owning /var/run/docker.sock
        DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)

        # Check if docker group already exists with the correct GID
        if ! getent group docker >/dev/null; then
            # Create docker group with the retrieved GID
            groupadd -g ${DOCKER_GID} docker
        else
            # If group exists but has a different GID, delete and recreate it
            current_gid=$(getent group docker | cut -d: -f3)
            if [ "$current_gid" != "$DOCKER_GID" ]; then
                groupdel docker
                groupadd -g ${DOCKER_GID} docker
            fi
        fi

        # Add airflow user to docker group
        usermod -aG docker airflow
        echo "User airflow added to docker group (GID: ${DOCKER_GID})"
    else
        echo "Warning: /var/run/docker.sock not found, skipping Docker configuration"
    fi
else
    echo "Not running as root, skipping Docker socket configuration"
fi

if [ -n "$AIRFLOW_ADMIN_PASSWORD" ]; then
    ADMIN_PWD="$AIRFLOW_ADMIN_PASSWORD"
else
    CONFIG_FILE="/opt/project/config.yaml"
    if [ -f "$CONFIG_FILE" ]; then
        ADMIN_PWD=$(python - <<'PY'
import yaml, sys, os
cfg = {}
try:
    with open(sys.argv[1], 'r') as f:
        cfg = yaml.safe_load(f) or {}
except Exception:
    pass
print(cfg.get('airflow', {}).get('admin_password', 'admin'))
PY
        "$CONFIG_FILE")
    else
        ADMIN_PWD="admin"
    fi
fi
export ADMIN_PWD

    set -e

    # Display Airflow version for debugging
    echo 'Checking Airflow installation:'
    su -c "airflow version" airflow

    # Function to wait for PostgreSQL to be ready
    wait_for_postgres() {
        local host="$1"
        local port="$2"
        local user="$3"
        local password="$4"
        local db="$5"
        local max_attempts=30
        local attempt=0

        echo "Waiting for PostgreSQL at $host:$port to be ready..."
        while [ $attempt -lt $max_attempts ]; do
            attempt=$((attempt+1))
            PGPASSWORD=$password psql -h "$host" -p "$port" -U "$user" -d "$db" -c 'SELECT 1' &>/dev/null && break
            echo "PostgreSQL not ready yet (attempt $attempt/$max_attempts)..."
            sleep 2
        done

        if [ $attempt -eq $max_attempts ]; then
            echo "Could not connect to PostgreSQL after $max_attempts attempts"
            exit 1
        fi

        echo "PostgreSQL is ready!"
    }

    # Extract connection information from connection string
    if [ -n "$AIRFLOW__DATABASE__SQL_ALCHEMY_CONN" ]; then
        conn_string=$AIRFLOW__DATABASE__SQL_ALCHEMY_CONN
        host=$(echo $conn_string | sed -n 's/.*@\([^:]*\).*/\1/p')
        port=$(echo $conn_string | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
        user=$(echo $conn_string | sed -n 's/.*:\/\/\([^:]*\).*/\1/p')
        password=$(echo $conn_string | sed -n 's/.*:\/\/[^:]*:\([^@]*\).*/\1/p')
        db=$(echo $conn_string | sed -n 's/.*\/\([^?]*\).*/\1/p')
        
        echo "Connecting to PostgreSQL at $host:$port as $user"
        wait_for_postgres "$host" "$port" "$user" "$password" "$db"
    fi

    # Initialize Airflow database
    echo 'Initializing Airflow database...'
    set +e
    su -c "/home/airflow/.local/bin/airflow db init" airflow
    db_init_exit_code=$?
    set -e
    echo "DB init command exited with code: $db_init_exit_code"
    if [ $db_init_exit_code -ne 0 ]; then
        echo "ERROR: airflow db init failed. Exiting."
        exit $db_init_exit_code
    fi

    # Create admin user if needed
    echo "Creating admin user if it doesn't exist..."
    set +e
    su -c "airflow users list --output table | awk '{print $2}' | grep -q '^admin$' || airflow users create --username admin --password "$ADMIN_PWD" --firstname Admin --lastname User --role Admin --email admin@example.com" airflow
    admin_user_exit_code=$?
    set -e
    echo "Admin user creation/check command exited with code: $admin_user_exit_code"
    if [ $admin_user_exit_code -ne 0 ]; then
        echo "ERROR: Failed to create or check admin user. Exiting."
        exit $admin_user_exit_code
    fi

    # Create airflow API user if needed
    echo "Creating airflow API user if it doesn't exist..."
    set +e # Désactiver temporairement l'arrêt sur erreur
    su -c "airflow users list --output table | awk '{print $2}' | grep -q '^airflow$' || airflow users create --username airflow --password airflow --firstname Airflow --lastname API --role Admin --email airflow@example.com" airflow
    exit_code=$?
    set -e # Réactiver l'arrêt sur erreur
    echo "User creation/check command for 'airflow' user exited with code: $exit_code"
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Failed to create or check airflow API user. Exiting."
        exit $exit_code
    fi

    # Créer l'utilisateur readonly en lecture seule si nécessaire
    echo "Creating readonly user if it doesn't exist..."
    set +e
    su -c "airflow users list --output table | awk '{print $2}' | grep -q '^readonly$' || airflow users create --username readonly --password readonly --firstname Read --lastname Only --role Viewer --email readonly@example.com" airflow
    readonly_exit_code=$?
    set -e
    echo "User creation/check command for 'readonly' user exited with code: $readonly_exit_code"
    if [ $readonly_exit_code -ne 0 ]; then
        echo "ERROR: Failed to create or check readonly user. Exiting."
        exit $readonly_exit_code
    fi

    echo 'Airflow initialization completed successfully!'

    # Exécuter la commande fournie en argument en tant qu'utilisateur airflow
    if [ "$#" -gt 0 ]; then
        echo "Executing command as airflow user: $@"
        # Utiliser le chemin complet pour les commandes Airflow
        if [ "$1" = "webserver" ]; then
            set -x # Enable tracing
            exec su -c "/home/airflow/.local/bin/airflow webserver" airflow
        elif [ "$1" = "scheduler" ]; then
            exec su -c "/home/airflow/.local/bin/airflow scheduler" airflow
        else
            # Pour les autres commandes, les passer telles quelles
            exec su -c "/home/airflow/.local/bin/airflow $*" airflow
        fi
    else
        echo "No command provided, exiting"
        exit 0
    fi
