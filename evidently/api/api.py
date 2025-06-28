import os
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from prometheus_client import start_http_server, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from scipy import stats
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths to data files
REFERENCE_DATA_PATH = '/app/reference/best_model_data.csv'
CURRENT_DATA_PATH = '/app/current/current_data.csv'

# Create FastAPI app
app = FastAPI(title="Evidently Data Drift API", 
              description="API to calculate data drift between reference and current datasets")

# Define Prometheus metrics
data_drift_score = Gauge('data_drift_score', 'Data Drift Score between reference and current data')
feature_drift_scores = {}  # Will be populated with feature-specific drift metrics

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
    return {
        "message": "Evidently Data Drift API is running",
        "endpoints": {
            "/drift": "Calculate and return drift metrics",
            "/metrics": "Prometheus metrics endpoint"
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
