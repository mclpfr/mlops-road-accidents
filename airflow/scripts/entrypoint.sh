#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Wrapper to run commands as airflow user if we are root
run_as_airflow() {
    if [ "$(id -u)" = "0" ]; then
        # If running as root, execute the command as the airflow user
        su -s /bin/bash -c "$*" airflow
    else
        # If already airflow user, just run the command
        bash -c "$*"
    fi
}

# Wrapper to exec commands as airflow user if we are root
exec_as_airflow() {
    if [ "$(id -u)" = "0" ]; then
        # If running as root, execute the command as the airflow user
        exec su -s /bin/bash -c "$*" airflow
    else
        # If already airflow user, just execute the command
        exec bash -c "$*"
    fi
}

# Ensure Airflow static assets are available in shared volume
if [ ! "$(ls -A /opt/airflow/www/static 2>/dev/null)" ]; then
    echo "Populating Airflow static assets into /opt/airflow/www/static..."
    STATIC_SRC=$(find /home/airflow/.local -name 'static' -type d -path '*/airflow/*' | head -n 1 || true)
    if [ -n "$STATIC_SRC" ]; then
        cp -r ${STATIC_SRC}/* /opt/airflow/www/static/ || echo "Failed to copy static assets"
        # Ensure readable permissions for nginx host
        chmod -R a+rX /opt/airflow/www/static || true
        echo "Static assets copied from ${STATIC_SRC}"
    else
        echo "Warning: Could not locate Airflow static assets directory."
    fi
else
    echo "Airflow static assets already present in /opt/airflow/www/static"
fi

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
    # Default admin password if not set in environment
    ADMIN_PWD="admin"
    
    # Try to read from config file if it exists
    CONFIG_FILE="/opt/project/config.yaml"
    if [ -f "$CONFIG_FILE" ]; then
        # Ensure the config file is readable by the airflow user
        chmod 644 "$CONFIG_FILE"

        echo "Loading Airflow configuration from $CONFIG_FILE..."
        TEMP_CONFIG_FILE="/tmp/airflow_config.sh"

        # Generate the config file as the airflow user and source it.
        if run_as_airflow "python3 /opt/airflow/scripts/load_config.py > $TEMP_CONFIG_FILE"; then
            . "$TEMP_CONFIG_FILE"
            rm -f "$TEMP_CONFIG_FILE"
        else
            echo "Warning: Failed to load configuration from $CONFIG_FILE. Using defaults."
        fi

        # Set ADMIN_PWD from the exported variable or use default
        ADMIN_PWD=${AIRFLOW__ADMIN_PASSWORD:-admin}
    else
        echo "Warning: Config file $CONFIG_FILE not found or not readable, using default admin password"
    fi
fi
export ADMIN_PWD

set -e

# Display Airflow version for debugging
echo 'Checking Airflow installation:'
run_as_airflow "airflow version"

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

# Initialize Airflow database with PostgreSQL
echo 'Initializing Airflow database with PostgreSQL...'
set +e

# Initialize the database
echo 'Running database migrations...'
run_as_airflow "airflow db migrate"
db_init_exit_code=$?
set -e

echo "DB migrate command exited with code: $db_init_exit_code"
if [ $db_init_exit_code -ne 0 ]; then
    echo "ERROR: airflow db migrate failed. Exiting."
    exit $db_init_exit_code
fi

# Create admin user if needed
echo "Creating admin user if it doesn't exist..."
set +e
run_as_airflow "airflow users list --output table | awk '{print \$2}' | grep -q '^admin$' || airflow users create --username admin --password '$ADMIN_PWD' --firstname Airflow --lastname Admin --role Admin --email admin@example.com"
admin_user_exit_code=$?
set -e
echo "Admin user creation/check command exited with code: $admin_user_exit_code"
if [ $admin_user_exit_code -ne 0 ]; then
    echo "ERROR: Failed to create or check admin user. Exiting."
    exit $admin_user_exit_code
fi

# Create airflow API user if needed
echo "Creating airflow API user if it doesn't exist..."
set +e # Temporarily disable exit on error
run_as_airflow "airflow users list --output table | awk '{print \$2}' | grep -q '^airflow$' || airflow users create --username airflow --password airflow --firstname Airflow --lastname API --role Admin --email airflow@example.com"
exit_code=$?
set -e # Re-enable exit on error
echo "User creation/check command for 'airflow' user exited with code: $exit_code"
if [ $exit_code -ne 0 ]; then
    echo "ERROR: Failed to create or check airflow API user. Exiting."
    exit $exit_code
fi

# Create a custom "Viewer" role with read-only permissions if needed
echo "Creating custom Viewer role if it doesn't exist..."
set +e
# Check if the role already exists
run_as_airflow "airflow roles list | grep -q 'Viewer'"
role_exists=$?
if [ $role_exists -ne 0 ]; then
    echo "Creating Viewer role with read-only permissions..."
    # Create the Viewer role
    run_as_airflow "airflow roles create Viewer"
    
    # Add read-only permissions
    run_as_airflow "airflow roles add-permissions Viewer menu.browse"
    run_as_airflow "airflow roles add-permissions Viewer menu.docs"
    run_as_airflow "airflow roles add-permissions Viewer menu.dags"
    run_as_airflow "airflow roles add-permissions Viewer dags.read"
    run_as_airflow "airflow roles add-permissions Viewer website.get_dag_runs"
    run_as_airflow "airflow roles add-permissions Viewer website.get_task_instances"
    run_as_airflow "airflow roles add-permissions Viewer website.get_dag"
    run_as_airflow "airflow roles add-permissions Viewer website.get_task"
    run_as_airflow "airflow roles add-permissions Viewer website.get_dag_code"
    run_as_airflow "airflow roles add-permissions Viewer website.get_task_logs"
    run_as_airflow "airflow roles add-permissions Viewer website.log"
else
    echo "Viewer role already exists."
fi
set -e

# Create/Update the 'readonly' user and assign the 'Viewer' role
echo "Checking for 'readonly' user..."
set +e

# Check if user exists
if ! run_as_airflow "airflow users list | grep -q 'readonly'"; then
    echo "User 'readonly' does not exist. Creating..."
    # Create the user directly with the Viewer role
    run_as_airflow "airflow users create --username readonly --password readonly --firstname Read --lastname Only --role Viewer --email readonly@example.com"
    echo "'readonly' user created successfully with 'Viewer' role."
else
    echo "User 'readonly' already exists. Ensuring 'Viewer' role is assigned."
    # Add the Viewer role. If the user already has the role, this command will fail.
    # We add '|| true' to ensure that this failure does not stop the script.
    run_as_airflow "airflow users add-role --username readonly --role Viewer" || true
    echo "'Viewer' role ensured for user 'readonly'."
fi

# Reset error checking
set -e
echo "'readonly' user setup complete."

echo 'Airflow initialization completed successfully!'

# Execute the command provided as argument as airflow user
if [ "$#" -gt 0 ]; then
    echo "Executing command as airflow user: $@"
    # Use the full path for Airflow commands
    if [ "$1" = "webserver" ]; then
        set -x # Enable tracing
        exec_as_airflow "airflow webserver"
    elif [ "$1" = "scheduler" ]; then
        exec_as_airflow "airflow scheduler"
    else
        # For other commands, pass them as is
        exec_as_airflow "airflow $*"
    fi
else
    echo "No command provided, exiting"
    exit 0
fi
