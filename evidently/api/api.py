import os
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Query
from prometheus_client import Gauge, generate_latest
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from scipy import stats
import pathlib
import requests
import json
import asyncio

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Paths ---
BASE_DIR = pathlib.Path(__file__).parent
REFERENCE_DATA_PATH = '/app/reference/best_model_data.csv'
CURRENT_DATA_PATH = '/app/current/current_data.csv'
DRIFT_STATUS_FILE = BASE_DIR / "drift_status.json"
STATIC_DIR = BASE_DIR / "static"

# --- FastAPI App Initialization ---
app = FastAPI(title="Evidently Data Drift API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- Prometheus Metrics ---
data_drift_score = Gauge('data_drift_score', 'Data Drift Score')
feature_drift_scores = {}

# --- State Management ---
def get_drift_state():
    if not DRIFT_STATUS_FILE.exists():
        return {"enabled": False, "percentage": 0.0}
    try:
        with open(DRIFT_STATUS_FILE, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return {"enabled": False, "percentage": 0.0}

def set_drift_state(enabled: bool, percentage: float):
    try:
        with open(DRIFT_STATUS_FILE, 'w') as f:
            json.dump({"enabled": enabled, "percentage": percentage}, f)
        logger.info(f"Drift state updated: enabled={enabled}, percentage={percentage}")
    except IOError as e:
        logger.error(f"Error writing drift status file: {e}")

# --- Core Logic ---
def load_data():
    try:
        reference_data = pd.read_csv(REFERENCE_DATA_PATH)
        current_data = pd.read_csv(CURRENT_DATA_PATH)
        return reference_data, current_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

def calculate_drift():
    try:
        reference_data, current_data = load_data()
        common_columns = list(set(reference_data.columns) & set(current_data.columns))
        
        overall_drift_score = 0.0
        num_drifted_features = 0
        feature_drift_results = {}
        
        drift_state = get_drift_state()
        artificial_drift_enabled = drift_state.get("enabled", False)
        artificial_drift_percentage = drift_state.get("percentage", 0.0)

        if artificial_drift_enabled:
            logger.info(f"Applying artificial drift of {artificial_drift_percentage*100}%")
        
        for column in common_columns:
            if not pd.api.types.is_numeric_dtype(reference_data[column]) or not pd.api.types.is_numeric_dtype(current_data[column]):
                continue
            
            ref_values = reference_data[column].dropna().values
            curr_values = current_data[column].dropna().values
            
            if len(ref_values) < 10 or len(curr_values) < 10:
                continue
            
            ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)
            drift_score = float(ks_stat)
            
            if artificial_drift_enabled:
                drift_score = min(1.0, drift_score + artificial_drift_percentage)
            
            is_drift = bool(p_value < 0.05)
            
            feature_drift_results[column] = {
                'drift_score': drift_score, 
                'p_value': float(p_value),
                'drift_detected': is_drift
            }
            
            if column not in feature_drift_scores:
                safe_name = column.lower().replace(" ", "_").replace("-", "_")
                feature_drift_scores[column] = Gauge(f'feature_drift_{safe_name}', f'Drift score for {column}')
            
            feature_drift_scores[column].set(drift_score)
            
            if is_drift:
                num_drifted_features += 1
            overall_drift_score += drift_score
        
        num_features = len(feature_drift_results)
        if num_features > 0:
            overall_drift_score /= num_features
            drift_detected = bool((num_drifted_features / num_features) > 0.3)
        else:
            drift_detected = False

        data_drift_score.set(overall_drift_score)
        logger.info(f"Overall data drift score: {overall_drift_score:.4f}, Drift detected: {drift_detected}")
        
        return {
            "drift_score": float(overall_drift_score),
            "drift_detected": drift_detected,
            "drifted_features_count": int(num_drifted_features),
            "total_features_analyzed": int(num_features),
            "feature_drift": feature_drift_results
        }
    except Exception as e:
        logger.error(f"Error calculating drift: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating drift: {str(e)}")

# --- Background Task ---
async def periodic_drift_calculation():
    while True:
        try:
            logger.info("Running scheduled drift calculation")
            calculate_drift()
        except Exception as e:
            logger.error(f"Error in scheduled drift calculation: {e}")
        await asyncio.sleep(10)

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Evidently Data Drift API")
    set_drift_state(False, 0.0)  # Reset drift state on startup
    asyncio.create_task(periodic_drift_calculation())

# --- API Endpoints ---
@app.get("/")
async def root():
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return FileResponse(str(html_file))
    return {"message": "Evidently Data Drift API is running"}

@app.get("/drift")
async def get_drift_endpoint():
    return calculate_drift()

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/force_drift")
async def force_drift(drift_percentage: float = Query(0.8, ge=0.0, le=1.0)):
    set_drift_state(True, drift_percentage)
    drift_result = calculate_drift()
    return {
        "message": f"Artificial drift of {drift_percentage*100}% has been applied",
        "drift_enabled": True,
        "drift_percentage": drift_percentage,
        "current_drift": drift_result
    }

@app.get("/reset_drift")
async def reset_drift():
    set_drift_state(False, 0.0)
    drift_result = calculate_drift()
    return {
        "message": "Artificial drift has been reset",
        "drift_enabled": False,
        "drift_percentage": 0.0,
        "current_drift": drift_result
    }

@app.get("/drift_status")
async def drift_status():
    return get_drift_state()

@app.post("/trigger_airflow_from_alert")
async def trigger_airflow_from_alert(request: Request):
    try:
        alert_data = await request.json()
        if not any(alert.get('labels', {}).get('alertname') == 'HighDataDrift' for alert in alert_data.get('alerts', [])):
            return JSONResponse(content={"status": "info", "message": "No high data drift alerts found"}, status_code=200)

        airflow_url = "http://airflow-webserver:8080/api/v1/dags/road_accidents/dagRuns"
        headers = {"Content-Type": "application/json", "Authorization": "Basic YWlyZmxvdzphaXJmbG93"}
        payload = {"conf": {"alert_data": alert_data, "triggered_by": "evidently_api"}}
        
        logger.info(f"Triggering Airflow DAG 'road_accidents'")
        response = requests.post(airflow_url, headers=headers, json=payload)
        
        if response.status_code in [200, 201]:
            logger.info(f"Successfully triggered Airflow DAG. Response: {response.json()}")
            return JSONResponse(content={"status": "success", "message": "Airflow DAG triggered", "airflow_response": response.json()})
        else:
            logger.error(f"Failed to trigger Airflow DAG. Status: {response.status_code}, Response: {response.text}")
            return JSONResponse(content={"status": "error", "message": f"Failed to trigger Airflow DAG: {response.text}"}, status_code=500)
            
    except Exception as e:
        logger.error(f"Error processing alert webhook: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
