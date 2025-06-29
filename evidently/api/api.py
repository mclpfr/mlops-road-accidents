import os
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import start_http_server, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response, FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from scipy import stats
import time
import random
import pathlib
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths to data files
REFERENCE_DATA_PATH = '/app/reference/best_model_data.csv'
CURRENT_DATA_PATH = '/app/current/current_data.csv'

# Create FastAPI app
app = FastAPI(title="Evidently Data Drift API", 
              description="API to calculate data drift between reference and current datasets")

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
static_dir = pathlib.Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Define Prometheus metrics
data_drift_score = Gauge('data_drift_score', 'Data Drift Score between reference and current data')
feature_drift_scores = {}  # Will be populated with feature-specific drift metrics

# Flag to track if artificial drift is enabled
ARTIFICIAL_DRIFT_ENABLED = False
ARTIFICIAL_DRIFT_PERCENTAGE = 0.0

# Initialize the app and metrics
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Evidently Data Drift API")
    # Start Prometheus metrics server
    start_http_server(8001)
    logger.info("Prometheus metrics server started on port 8001")
    
    # Initialize drift calculation on startup
    try:
        calculate_drift()
    except Exception as e:
        logger.error(f"Error during initial drift calculation: {e}")

def load_data():
    """Load reference and current data from CSV files."""
    try:
        logger.info(f"Loading reference data from {REFERENCE_DATA_PATH}")
        reference_data = pd.read_csv(REFERENCE_DATA_PATH)
        
        logger.info(f"Loading current data from {CURRENT_DATA_PATH}")
        current_data = pd.read_csv(CURRENT_DATA_PATH)
        
        return reference_data, current_data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

def calculate_drift():
    """Calculate data drift between reference and current datasets using statistical methods."""
    try:
        reference_data, current_data = load_data()
        
        logger.info("Calculating data drift...")
        
        # Get common columns between the two datasets
        common_columns = list(set(reference_data.columns) & set(current_data.columns))
        logger.info(f"Found {len(common_columns)} common columns for drift analysis")
        
        # Initialize drift metrics
        overall_drift_score = 0.0
        drift_detected = False
        feature_drift_results = {}
        num_drifted_features = 0
        
        # Apply artificial drift if enabled
        global ARTIFICIAL_DRIFT_ENABLED, ARTIFICIAL_DRIFT_PERCENTAGE
        if ARTIFICIAL_DRIFT_ENABLED:
            logger.info(f"Applying artificial drift of {ARTIFICIAL_DRIFT_PERCENTAGE*100}%")
        
        # Calculate drift for each feature
        for column in common_columns:
            try:
                # Skip non-numeric columns for simplicity
                if not pd.api.types.is_numeric_dtype(reference_data[column]) or \
                   not pd.api.types.is_numeric_dtype(current_data[column]):
                    continue
                
                # Get values for the column from both datasets
                ref_values = reference_data[column].dropna().values
                curr_values = current_data[column].dropna().values
                
                # Skip if not enough data
                if len(ref_values) < 10 or len(curr_values) < 10:
                    continue
                
                # Calculate KS test for distribution comparison
                ks_stat, p_value = stats.ks_2samp(ref_values, curr_values)
                
                # Normalize KS statistic to a 0-1 scale as drift score
                # KS statistic is already between 0-1, where 0 means identical distributions
                # and 1 means completely different distributions
                drift_score = ks_stat
                
                # Apply artificial drift if enabled
                if ARTIFICIAL_DRIFT_ENABLED:
                    # Artificially increase the drift score based on the configured percentage
                    drift_score = min(1.0, drift_score + ARTIFICIAL_DRIFT_PERCENTAGE)
                
                # Determine if drift is detected (p-value < 0.05 indicates statistical significance)
                is_drift = p_value < 0.05
                
                # Store feature drift results
                feature_drift_results[column] = {
                    'drift_score': drift_score,
                    'p_value': p_value,
                    'drift_detected': is_drift
                }
                
                # Update Prometheus metrics for this feature
                if column not in feature_drift_scores:
                    safe_name = column.lower().replace(" ", "_").replace("-", "_")
                    feature_drift_scores[column] = Gauge(
                        f'feature_drift_{safe_name}', 
                        f'Drift score for feature {column}'
                    )
                
                # Set the value
                feature_drift_scores[column].set(drift_score)
                
                # Log feature drift
                logger.info(f"Feature {column} drift score: {drift_score:.4f}, p-value: {p_value:.4f}, drift detected: {is_drift}")
                
                # Count drifted features and add to overall score
                if is_drift:
                    num_drifted_features += 1
                overall_drift_score += drift_score
                
            except Exception as e:
                logger.warning(f"Error calculating drift for feature {column}: {e}")
        
        # Calculate overall drift metrics
        num_features = len(feature_drift_results)
        if num_features > 0:
            # Average drift score across all features
            overall_drift_score = overall_drift_score / num_features
            
            # If more than 30% of features have drift, consider it significant
            drift_detected = (num_drifted_features / num_features) > 0.3
        
        # Update Prometheus metrics
        data_drift_score.set(overall_drift_score)
        
        # Log overall drift information
        logger.info(f"Overall data drift score: {overall_drift_score:.4f}, Drift detected: {drift_detected}")
        logger.info(f"Number of drifted features: {num_drifted_features} out of {num_features}")
        
        return {
            "drift_score": overall_drift_score,
            "drift_detected": drift_detected,
            "drifted_features_count": num_drifted_features,
            "total_features_analyzed": num_features,
            "feature_drift": feature_drift_results
        }
    except Exception as e:
        logger.error(f"Error calculating drift: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating drift: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with basic API info."""
    # Return the HTML page for drift control
    html_file = pathlib.Path(__file__).parent / "static" / "index.html"
    if html_file.exists():
        return FileResponse(str(html_file))
    else:
        # Fallback to JSON response if HTML file doesn't exist
        return {
            "message": "Evidently Data Drift API is running",
            "endpoints": {
                "/drift": "Calculate and return drift metrics",
                "/metrics": "Prometheus metrics endpoint",
                "/force_drift": "Force artificial drift",
                "/reset_drift": "Reset artificial drift",
                "/drift_status": "Get current drift status"
            }
        }

@app.get("/drift")
async def get_drift():
    """Calculate and return drift metrics."""
    return calculate_drift()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/force_drift")
async def force_drift(drift_percentage: float = 0.8):
    """Force an artificial drift with the specified percentage (0.0 to 1.0)."""
    global ARTIFICIAL_DRIFT_ENABLED, ARTIFICIAL_DRIFT_PERCENTAGE
    
    # Validate input
    if drift_percentage < 0.0 or drift_percentage > 1.0:
        raise HTTPException(status_code=400, detail="Drift percentage must be between 0.0 and 1.0")
    
    # Enable artificial drift
    ARTIFICIAL_DRIFT_ENABLED = True
    ARTIFICIAL_DRIFT_PERCENTAGE = drift_percentage
    
    logger.info(f"Artificial drift enabled with {drift_percentage*100}% drift")
    
    # Recalculate drift immediately
    drift_result = calculate_drift()
    
    return {
        "message": f"Artificial drift of {drift_percentage*100}% has been applied",
        "drift_enabled": ARTIFICIAL_DRIFT_ENABLED,
        "drift_percentage": ARTIFICIAL_DRIFT_PERCENTAGE,
        "current_drift": drift_result
    }

@app.post("/reset_drift")
async def reset_drift():
    """Reset any artificial drift to normal operation."""
    global ARTIFICIAL_DRIFT_ENABLED, ARTIFICIAL_DRIFT_PERCENTAGE
    
    # Disable artificial drift
    ARTIFICIAL_DRIFT_ENABLED = False
    ARTIFICIAL_DRIFT_PERCENTAGE = 0.0
    
    logger.info("Artificial drift has been reset")
    
    # Recalculate drift immediately
    drift_result = calculate_drift()
    
    return {
        "message": "Artificial drift has been reset to normal operation",
        "drift_enabled": ARTIFICIAL_DRIFT_ENABLED,
        "drift_percentage": ARTIFICIAL_DRIFT_PERCENTAGE,
        "current_drift": drift_result
    }

@app.get("/drift_status")
async def drift_status():
    """Get the current status of artificial drift."""
    return {
        "drift_enabled": ARTIFICIAL_DRIFT_ENABLED,
        "drift_percentage": ARTIFICIAL_DRIFT_PERCENTAGE
    }

@app.post("/trigger_airflow_from_alert")
async def trigger_airflow_from_alert(request: Request):
    """Endpoint to receive alerts from AlertManager and trigger Airflow DAG."""
    try:
        # Get the alert data from the request
        alert_data = await request.json()
        logger.info(f"Received alert from AlertManager: {alert_data}")
        
        # Extract relevant information from the alert
        alerts = alert_data.get('alerts', [])
        if not alerts:
            logger.warning("No alerts found in the request")
            return JSONResponse(content={"status": "error", "message": "No alerts found"}, status_code=400)
        
        # Check if any of the alerts is for high data drift
        high_drift_alerts = [alert for alert in alerts if alert.get('labels', {}).get('alertname') == 'HighDataDrift']
        if not high_drift_alerts:
            logger.warning("No high data drift alerts found")
            return JSONResponse(content={"status": "error", "message": "No high data drift alerts found"}, status_code=400)
        
        # Trigger the Airflow DAG
        airflow_url = "http://airflow-webserver:8080/api/v1/dags/road_accidents/dagRuns"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Basic YWlyZmxvdzphaXJmbG93"  # Base64 encoded airflow:airflow
        }
        payload = {
            "conf": {
                "alert_data": alert_data,
                "triggered_by": "evidently_api"
            }
        }
        
        logger.info(f"Triggering Airflow DAG 'road_accidents' with payload: {payload}")
        
        # Make the request to Airflow API
        response = requests.post(airflow_url, headers=headers, json=payload)
        
        if response.status_code in [200, 201]:
            logger.info(f"Successfully triggered Airflow DAG. Response: {response.json()}")
            return JSONResponse(content={
                "status": "success", 
                "message": "Airflow DAG triggered successfully",
                "airflow_response": response.json()
            })
        else:
            logger.error(f"Failed to trigger Airflow DAG. Status code: {response.status_code}, Response: {response.text}")
            return JSONResponse(content={
                "status": "error", 
                "message": f"Failed to trigger Airflow DAG: {response.text}"
            }, status_code=500)
            
    except Exception as e:
        logger.error(f"Error processing alert webhook: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

# Schedule periodic drift calculation (every 5 minutes)
@app.on_event("startup")
async def schedule_drift_calculation():
    import asyncio
    
    async def periodic_drift_calculation():
        while True:
            try:
                logger.info("Running scheduled drift calculation")
                calculate_drift()
            except Exception as e:
                logger.error(f"Error in scheduled drift calculation: {e}")
            
            # Wait for 5 minutes
            await asyncio.sleep(300)
    
    # Start the periodic task
    asyncio.create_task(periodic_drift_calculation())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
