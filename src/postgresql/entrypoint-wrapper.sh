#!/bin/sh
# This script acts as a wrapper for the original postgres entrypoint.
# It reads database configuration from a YAML file, exports it as environment variables,
# and then executes the original entrypoint.

set -e

# The path to the config file, mounted into the container.
CONFIG_FILE="/app/config.yaml"

# Check if the config file exists before proceeding.
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at $CONFIG_FILE" >&2
    exit 1
fi

# Use yq to parse the YAML file and export the variables.
# The '-r' flag outputs raw strings without quotes.
export POSTGRES_USER=$(yq -r '.postgresql.user' "$CONFIG_FILE")
export POSTGRES_PASSWORD=$(yq -r '.postgresql.password' "$CONFIG_FILE")
export POSTGRES_DB=$(yq -r '.postgresql.database' "$CONFIG_FILE")

# Execute the original Docker entrypoint script, passing along any arguments
# that were passed to this script. This will start the postgres server.
exec /usr/local/bin/docker-entrypoint.sh "$@"
