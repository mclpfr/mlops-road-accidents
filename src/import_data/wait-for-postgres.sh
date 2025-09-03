#!/bin/bash

set -e

# Args remain supported but can be overridden by env vars
host_arg="$1"
port_arg="$2"
shift 2 || true
cmd="$@"

# Prefer explicit env vars if present
POSTGRES_HOST="${POSTGRES_HOST:-$host_arg}"
POSTGRES_PORT="${POSTGRES_PORT:-$port_arg}"
POSTGRES_USER="${POSTGRES_USER:-}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}"
POSTGRES_DB="${POSTGRES_DB:-}"

# Fallback to config.yaml when needed
if [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ] || [ -z "$POSTGRES_DB" ]; then
  CONFIG_PATH="${CONFIG_PATH:-/app/config.yaml}"
  read cfg_user cfg_pass cfg_db < <(
  python - <<'PY'
import os, yaml
cfg_path = os.environ.get('CONFIG_PATH', '/app/config.yaml')
with open(cfg_path) as f:
    data = yaml.safe_load(f) or {}
pg = data.get('postgresql', {})
print(pg.get('user', ''), pg.get('password', ''), pg.get('database', 'postgres'))
PY
  )
  POSTGRES_USER="${POSTGRES_USER:-$cfg_user}"
  POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$cfg_pass}"
  POSTGRES_DB="${POSTGRES_DB:-$cfg_db}"
fi

if [ -z "$POSTGRES_HOST" ] || [ -z "$POSTGRES_PORT" ] || [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ] || [ -z "$POSTGRES_DB" ]; then
  echo "[wait-for-postgres] Missing connection info. Host=$POSTGRES_HOST Port=$POSTGRES_PORT User=$POSTGRES_USER DB=$POSTGRES_DB" >&2
  exit 1
fi

export PGPASSWORD="$POSTGRES_PASSWORD"

echo "[wait-for-postgres] Trying to connect: host=$POSTGRES_HOST port=$POSTGRES_PORT user=$POSTGRES_USER db=$POSTGRES_DB" >&2

# Retry until PostgreSQL accepts connections; print the error for diagnostics
until psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q' >/dev/null 2> >(err=$(cat); echo "[wait-for-postgres] psql error: $err" >&2); do
  >&2 echo "[wait-for-postgres] PostgreSQL is not available - waiting..."
  sleep 1
done

>&2 echo "[wait-for-postgres] PostgreSQL is ready - executing command"
exec $cmd