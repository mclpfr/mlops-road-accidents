#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Vérifier si l'utilisateur actuel est root
if [ "$(id -u)" = "0" ]; then
    # Configurer le groupe docker pour l'accès au socket (en tant que root)
    echo "Configuring Docker socket access..."
    if [ -S /var/run/docker.sock ]; then
        # Récupérer le GID du groupe propriétaire de /var/run/docker.sock
        DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)

        # Vérifier si le groupe docker existe déjà avec le bon GID
        if ! getent group docker >/dev/null; then
            # Créer le groupe docker avec le GID récupéré
            groupadd -g ${DOCKER_GID} docker
        else
            # Si le groupe existe mais a un GID différent, le supprimer et le recréer
            current_gid=$(getent group docker | cut -d: -f3)
            if [ "$current_gid" != "$DOCKER_GID" ]; then
                groupdel docker
                groupadd -g ${DOCKER_GID} docker
            fi
        fi

        # Ajouter l'utilisateur airflow au groupe docker
        usermod -aG docker airflow
        echo "User airflow added to docker group (GID: ${DOCKER_GID})"
    else
        echo "Warning: /var/run/docker.sock not found, skipping Docker configuration"
    fi
else
    echo "Not running as root, skipping Docker socket configuration"
fi

# Ne pas changer d'utilisateur, exécuter en tant que root
# Cela est cohérent avec notre configuration dans docker-compose.yml

    set -e

    # Afficher la version d'Airflow pour débogage
    echo 'Checking Airflow installation:'
    su -c "airflow version" airflow

    # Fonction pour attendre que PostgreSQL soit prêt
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

    # Extraire les informations de connexion de la chaîne de connexion
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

    # Initialiser la base de données Airflow
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

    # Créer un utilisateur administrateur si nécessaire
    echo "Creating admin user if it doesn't exist..."
    set +e
    su -c "airflow users list | grep -q '^admin$' || airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com" airflow
    admin_user_exit_code=$?
    set -e
    echo "Admin user creation/check command exited with code: $admin_user_exit_code"
    if [ $admin_user_exit_code -ne 0 ]; then
        echo "ERROR: Failed to create or check admin user. Exiting."
        exit $admin_user_exit_code
    fi

    # Créer l'utilisateur airflow pour l'API si nécessaire
    echo "Creating airflow API user if it doesn't exist..."
    set +e # Désactiver temporairement l'arrêt sur erreur
    su -c "airflow users list | grep -q '^airflow$' || airflow users create --username airflow --password airflow --firstname Airflow --lastname API --role Admin --email airflow@example.com" airflow
    exit_code=$?
    set -e # Réactiver l'arrêt sur erreur
    echo "User creation/check command for 'airflow' user exited with code: $exit_code"
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Failed to create or check airflow API user. Exiting."
        exit $exit_code
    fi

    echo 'Airflow initialization completed successfully!'

    # Exécuter la commande fournie en argument en tant qu'utilisateur airflow
    if [ "$#" -gt 0 ]; then
        echo "Executing command as airflow user: $@"
        # Utiliser le chemin complet pour les commandes Airflow
        if [ "$1" = "webserver" ]; then
            set -x # Activer le traçage
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