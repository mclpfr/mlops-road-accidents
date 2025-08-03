"""Utilities to query problems stored in PostgreSQL logs table."""
import os
import datetime as dt
from typing import List, Dict

import psycopg2

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "")
POSTGRES_USER = os.getenv("POSTGRES_USER", "")
import yaml
from pathlib import Path

POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
if not (POSTGRES_HOST and POSTGRES_DB and POSTGRES_USER and POSTGRES_PASSWORD):
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
            POSTGRES_DB = POSTGRES_DB or pg_cfg.get("database", "road_accidents")
        except Exception as e:
            print(f"⚠️ Impossible de lire config.yaml pour le mot de passe PG: {e}")

_DEF_CONN = None

def _get_conn():
    global _DEF_CONN
    if _DEF_CONN is None or _DEF_CONN.closed:
        kwargs = dict(host=POSTGRES_HOST, dbname=POSTGRES_DB, user=POSTGRES_USER)
        if 'POSTGRES_PORT' in globals():
            kwargs['port'] = POSTGRES_PORT
        if POSTGRES_PASSWORD:
            kwargs["password"] = POSTGRES_PASSWORD
        _DEF_CONN = psycopg2.connect(**kwargs)
    return _DEF_CONN


def recent_problems(hours: int = 24) -> Dict:
    """Return aggregated problems within last `hours`."""
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT message, COUNT(*) AS cnt
            FROM logs
            WHERE timestamp >= NOW() - INTERVAL '%s hours'
            GROUP BY message
            ORDER BY cnt DESC
            LIMIT 20;
            """,
            (hours,)
        )
        rows = cur.fetchall()
        total = sum(r[1] for r in rows)
        top = [
            {"message": r[0][:120], "count": r[1], "is_urgent": r[1] >= 3}
            for r in rows
        ]
    return {"total_errors": total, "top_issues": top}
