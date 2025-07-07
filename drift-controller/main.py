import os
import logging
import requests
import asyncio
from fastapi import FastAPI, HTTPException, Response
from prometheus_api_client import PrometheusConnect
from prometheus_client import Gauge, CONTENT_TYPE_LATEST, generate_latest

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://airflow-webserver:8080/api/v1/dags/road_accidents/dagRuns")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "airflow")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "airflow")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.5"))
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "60"))

try:
    prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)
except Exception as e:
    logger.error(f"Failed to connect to Prometheus: {e}")
    prom = None

app = FastAPI(title="Drift Controller Service")

# --- Prometheus Exporter ---
# This gauge will be scraped by Prometheus (e.g., http://drift-controller:8002/metrics)
drift_gauge = Gauge('data_drift_score', 'Current data-drift score')

# --- State ---
is_pipeline_running = False  # True when a retraining DAG run is active


def refresh_pipeline_state():
    """Query Airflow to detect whether a retraining DAG run is still active."""
    global is_pipeline_running
    try:
        # List running dagRuns (Airflow REST: /dags/{dag_id}/dagRuns?state=running)
        resp = requests.get(
            AIRFLOW_URL,
            auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
            params={"state": "running", "limit": 1},
            timeout=15,
        )
        resp.raise_for_status()
        active = resp.json().get("total_entries", 0) > 0
        if is_pipeline_running and not active:
            logger.info("Detected that retraining DAG run has finished – allowing future triggers.")
        is_pipeline_running = active
    except Exception as e:
        logger.error(f"Unable to refresh pipeline state from Airflow: {e}")

# --- Core Logic ---
def check_drift_and_trigger():
    global is_pipeline_running
    logger.info("Running drift check...")

    if not prom:
        logger.error("Prometheus client not available. Skipping check.")
        return

    if is_pipeline_running:
        logger.info("Pipeline re-training is already in progress. Skipping check.")
        return

    try:
        metric_data = prom.get_current_metric_value(metric_name='data_drift_score')
        if not metric_data:
            logger.warning("`data_drift_score` metric not found in Prometheus.")
            return

        drift_score = float(metric_data[0]['value'][1])
        logger.info(f"Current data drift score: {drift_score:.4f}")
        drift_gauge.set(drift_score)

        if drift_score > DRIFT_THRESHOLD:
            logger.warning(f"Drift score {drift_score:.4f} exceeds threshold {DRIFT_THRESHOLD}. Triggering Airflow DAG.")
            trigger_airflow_dag()

    except Exception as e:
        logger.error(f"An error occurred during drift check: {e}", exc_info=True)

def trigger_airflow_dag():
    global is_pipeline_running
    logger.info(f"Attempting to trigger Airflow DAG 'road_accidents'")
    try:
        response = requests.post(
            AIRFLOW_URL,
            auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
            json={"conf": {}},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        logger.info(f"Successfully triggered Airflow DAG. Status: {response.status_code}, Response: {response.json()}")
        is_pipeline_running = True
    except requests.exceptions.HTTPError as e:
        logger.error(f"Failed to trigger Airflow DAG. Status: {e.response.status_code}, Response: {e.response.text}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while triggering Airflow DAG: {e}", exc_info=True)

# --- Background Task ---
async def run_periodic_drift_check():
    logger.info("Starting periodic drift check task.")
    while True:
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)
        try:
            refresh_pipeline_state()
            check_drift_and_trigger()
        except Exception as e:
            logger.error(f"Error during periodic drift check: {e}", exc_info=True)

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    logger.info("Scheduling periodic drift check.")
    asyncio.create_task(run_periodic_drift_check())

@app.get("/metrics", summary="Prometheus Metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", summary="Health Check")
def read_root():
    return {"status": "ok", "message": "Drift Controller is running"}

@app.post("/reset", summary="Reset Pipeline State")
def reset_pipeline_state():
    global is_pipeline_running
    is_pipeline_running = False
    drift_gauge.set(0)
    logger.info("Pipeline state has been manually reset and drift score forced to 0.")
    return {"status": "ok", "message": "Pipeline state reset."}
