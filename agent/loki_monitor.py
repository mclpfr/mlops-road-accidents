"""Background task to periodically pull error/warning logs from Loki and store them in PostgreSQL."""

import os
import asyncio
import datetime as dt
from typing import List, Dict

import aiohttp
import psycopg2
from psycopg2.extras import execute_values

LOKI_URL = os.getenv("LOKI_URL", "http://loki:3100")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "")
POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

import yaml

# Retention in days (default 30) can be overridden via env or config.yaml
RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "30"))  # default 30
from pathlib import Path

if not (POSTGRES_HOST and POSTGRES_PORT and POSTGRES_DB and POSTGRES_USER and POSTGRES_PASSWORD):
    cfg_path = Path("/app/config.yaml")
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            pg_cfg = cfg.get("postgresql", {})
            POSTGRES_HOST = POSTGRES_HOST or pg_cfg.get("host", "postgres_service")
            POSTGRES_PORT = pg_cfg.get("port", "5432")
            POSTGRES_USER = POSTGRES_USER or pg_cfg.get("user", "postgres")
            POSTGRES_PASSWORD = POSTGRES_PASSWORD or pg_cfg.get("password", "")
            # Optional log retention key
            if not os.getenv("LOG_RETENTION_DAYS"):
                RETENTION_DAYS = int(cfg.get("log_retention_days", RETENTION_DAYS))
            POSTGRES_DB = POSTGRES_DB or pg_cfg.get("database", "road_accidents")
        except Exception as e:
            print(f"⚠️ Impossible de lire config.yaml pour le mot de passe PG: {e}")

TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS logs (
  id SERIAL PRIMARY KEY,
  timestamp TIMESTAMPTZ,
  level TEXT,
  container_name TEXT,
  service_name TEXT,
  message TEXT,
  inserted_at TIMESTAMPTZ DEFAULT NOW()
);
"""

# LogQL query: select all logs then regex filter for error/warning (case-insensitive)
QUERY = '{container=~".+"} |~ "(?i)(error|warning)"'

async def _fetch_loki_logs(session: aiohttp.ClientSession, since: int) -> List[Dict]:
    """Fetch logs newer than `since` (unix ns) matching error/warning."""
    params = {
        "query": QUERY,
        "start": since,
        "limit": 5000,
    }
    async with session.get(f"{LOKI_URL}/loki/api/v1/query_range", params=params, timeout=30) as resp:
        resp.raise_for_status()
        data = await resp.json()
    streams = data.get("data", {}).get("result", [])
    logs: List[Dict] = []
    for stream in streams:
        labels = stream.get("stream", {})
        values = stream.get("values", [])
        for ts_ns, msg in values:
            ts = dt.datetime.fromtimestamp(int(ts_ns) / 1e9, dt.timezone.utc)
            level = labels.get("level") or ("error" if "error" in msg.lower() else "warning")
            container = labels.get("container") or labels.get("container_name")
            service = labels.get("service")
            logs.append({
                "timestamp": ts,
                "level": level,
                "container_name": container,
                "service_name": service,
                "message": msg[:1000],  # safety limit
            })
    return logs


def _insert_logs(conn, rows: List[Dict]):
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO logs (timestamp, level, container_name, service_name, message) VALUES %s ON CONFLICT DO NOTHING",
            [(
                r["timestamp"],
                r["level"],
                r.get("container_name"),
                r.get("service_name"),
                r["message"],
            ) for r in rows],
        )
    conn.commit()

async def monitor_loop(interval: int = 300):
    """Main coroutine that runs forever pulling logs and storing to PG."""
    last_ts_ns = int((dt.datetime.utcnow() - dt.timedelta(minutes=10)).timestamp() * 1e9)
    conn_kwargs = dict(host=POSTGRES_HOST, dbname=POSTGRES_DB, user=POSTGRES_USER)
    if 'POSTGRES_PORT' in globals():
        conn_kwargs['port'] = POSTGRES_PORT
    if POSTGRES_PASSWORD:
        conn_kwargs["password"] = POSTGRES_PASSWORD
    conn = psycopg2.connect(**conn_kwargs)
    with conn.cursor() as cur:
        cur.execute(TABLE_SCHEMA)
    conn.commit()

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                logs = await _fetch_loki_logs(session, since=last_ts_ns)
                print(f"[loki_monitor] fetched {len(logs)} logs")
                if logs:
                    last_ts_ns = int(max(l["timestamp"] for l in logs).timestamp() * 1e9) + 1
                    _insert_logs(conn, logs)
                    print(f"[loki_monitor] inserted {len(logs)} logs into PG")

                # Purge old logs once a day after insertion batch
                try:
                    cutoff_ts = dt.datetime.utcnow() - dt.timedelta(days=RETENTION_DAYS)
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM logs WHERE timestamp < %s", (cutoff_ts,))
                    conn.commit()
                except Exception as e:
                    print(f"[loki_monitor] purge error: {e}")
            except Exception as e:
                print(f"[loki_monitor] error: {e}")
            await asyncio.sleep(interval)
