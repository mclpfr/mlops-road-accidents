#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if config.yaml exists
if [ ! -f /app/config.yaml ]; then
    echo "Error: /app/config.yaml not found!" >&2
    exit 1
fi

# Read values from config.yaml and export them as environment variables for Grafana
export GF_SERVER_ROOT_URL=$(yq -r '.grafana.root_url' /app/config.yaml)
export GF_SERVER_DOMAIN=$(yq -r '.grafana.domain' /app/config.yaml)
export GF_SECURITY_ADMIN_USER=$(yq -r '.grafana.admin_user' /app/config.yaml)
export GF_SECURITY_ADMIN_PASSWORD=$(yq -r '.grafana.admin_password' /app/config.yaml)

# Check if variables were read correctly
if [ -z "$GF_SERVER_ROOT_URL" ] || [ -z "$GF_SERVER_DOMAIN" ] || [ -z "$GF_SECURITY_ADMIN_USER" ] || [ -z "$GF_SECURITY_ADMIN_PASSWORD" ]; then
    echo "Error: Could not read one or more required Grafana variables from /app/config.yaml." >&2
    exit 1
fi

# Execute the original Grafana entrypoint
exec /run.sh
