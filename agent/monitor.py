"""Lightweight autonomous monitoring daemon.

Usage:
    python monitor.py

The script runs indefinitely, scanning Docker containers every INTERVAL_SECONDS.
It writes JSON lines describing any action taken to a shared file (EVENTS_PATH)
so that the Streamlit dashboard can pick them up in near real-time.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import docker

# Configuration ----------------------------------------------------------------
INTERVAL = int(os.getenv("INTERVAL_SECONDS", "30"))
CPU_THRESHOLD = float(os.getenv("CPU_THRESHOLD", "90"))  # %
MEM_THRESHOLD = float(os.getenv("MEM_THRESHOLD", "90"))  # % of limit if limit set, else host
EVENTS_PATH = Path(os.getenv("AGENT_EVENTS_PATH", "/tmp/agent_events.jsonl"))
MAX_RESTARTS = int(os.getenv("MAX_RESTARTS", "3"))

# Only containers with this label are considered always-on services
ALWAYS_ON_LABEL = os.getenv("ALWAYS_ON_LABEL", "com.company.always_on")

client = docker.from_env()

# Keep restart counters per container per 10-minute window
restart_counters: dict[str, list[datetime]] = defaultdict(list)

def write_event(container: str, action: str, status: str, message: str) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "container": container,
        "action": action,
        "status": status,
        "message": message,
    }
    with EVENTS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def cpu_mem_usage(stats: dict) -> tuple[float, float]:
    """Return (cpu%, mem%) from Docker API stats payload."""
    cpu_delta = (
        stats["cpu_stats"]["cpu_usage"]["total_usage"]
        - stats["precpu_stats"]["cpu_usage"]["total_usage"]
    )
    system_delta = stats["cpu_stats"].get("system_cpu_usage", 0) - stats["precpu_stats"].get(
        "system_cpu_usage", 0
    )
    cpu_pct = 0.0
    if system_delta > 0:
        cpu_pct = (
            cpu_delta / system_delta * stats["cpu_stats"].get("online_cpus", 1) * 100.0
        )

    mem_usage = stats["memory_stats"].get("usage", 0)
    mem_limit = stats["memory_stats"].get("limit", 1)
    mem_pct = mem_usage / mem_limit * 100.0 if mem_limit else 0.0
    return cpu_pct, mem_pct


def should_restart(name: str) -> bool:
    """Rate-limit restarts per container to MAX_RESTARTS per 10-minute window."""
    now = datetime.utcnow()
    window = now - timedelta(minutes=10)
    # prune old timestamps
    restart_counters[name] = [ts for ts in restart_counters[name] if ts > window]
    if len(restart_counters[name]) >= MAX_RESTARTS:
        return False
    restart_counters[name].append(now)
    return True


def monitor_once() -> None:
    for c in client.containers.list(all=True):
        name = c.name
        labels = c.labels or {}
        always_on = labels.get(ALWAYS_ON_LABEL) == "true"
        state = c.attrs["State"]
        status = state["Status"]  # running, exited, etc.
        exit_code = state.get("ExitCode", 0)
        health = state.get("Health", {}).get("Status")  # healthy/unhealthy/starting/None

        incident = None
        if always_on:
            if status != "running":
                if exit_code != 0:
                    incident = f"Exited with code {exit_code}"
                else:
                    incident = "Exited unexpectedly"
            elif health == "unhealthy":
                incident = "Healthcheck failed"
        # CPU/MEM
        # Resource usage checks apply only to always-on services
        if status == "running" and always_on:
            try:
                stats = c.stats(stream=False)
                cpu_pct, mem_pct = cpu_mem_usage(stats)
                if cpu_pct > CPU_THRESHOLD:
                    incident = f"CPU {cpu_pct:.1f}%>Thresh"
                elif mem_pct > MEM_THRESHOLD:
                    incident = f"MEM {mem_pct:.1f}%>Thresh"
            except Exception:
                pass

        if incident:
            msg = incident
            if always_on:
                try:
                    if should_restart(name):
                        c.restart()
                        action_taken = "restart"
                        msg += "; restarted"
                    else:
                        action_taken = "alert"
                        msg += "; restart limit reached"
                except Exception as exc:
                    action_taken = "error"
                    msg += f"; restart failed: {exc}"
            else:
                # Non always-on containers: only alert, no automated restart
                action_taken = "alert"

            write_event(name, action_taken, status, msg)


def main() -> None:
    print(f"🩺 Monitoring daemon started (interval={INTERVAL}s)…", flush=True)
    try:
        while True:
            monitor_once()
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("Interrupted, exiting…", file=sys.stderr)


if __name__ == "__main__":
    main()
