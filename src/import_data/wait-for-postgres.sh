#!/bin/bash

set -e

host="$1"
port="$2"
shift 2
cmd="$@"

CONFIG_PATH="${CONFIG_PATH:-/app/config.yaml}"
# Extract credentials via Python (reliable YAML parsing)
read POSTGRES_USER POSTGRES_PASSWORD < <(
python - <<'PY'
import os, yaml, sys
cfg_path = os.environ.get('CONFIG_PATH', '/app/config.yaml')
with open(cfg_path) as f:
    data = yaml.safe_load(f)
pg = data.get('postgresql', {})
print(pg.get('user', ''), pg.get('password', ''))
PY
)

if [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ]; then
  echo "Could not extract Postgres credentials from $CONFIG_PATH" >&2
  exit 1
fi

export PGPASSWORD="$POSTGRES_PASSWORD"

until psql -h "$host" -p "$port" -U "$POSTGRES_USER" -d "postgres" -c '\q' >/dev/null 2>&1; do
  >&2 echo "PostgreSQL is not available - waiting..."
  sleep 1
done

>&2 echo "PostgreSQL is ready - executing command"
exec $cmd 