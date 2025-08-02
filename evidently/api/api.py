import os
import yaml
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Query
from prometheus_client import Gauge, generate_latest
from fastapi.responses import Response, FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from scipy import stats
import pathlib
import requests
import json
import shutil
import asyncio
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.figure import Figure
from datetime import datetime
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

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
# Load config
with open(BASE_DIR / 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)
evidently_config = config.get('evidently', {})

app = FastAPI(
    title="Evidently Data Drift API",
    root_path=evidently_config.get('url_prefix', os.getenv('EVIDENTLY_URL_PREFIX', ''))
)

# Middleware pour le reverse proxy
if os.getenv('EVIDENTLY_ENABLE_PROXY_FIX', 'False').lower() == 'true':
    from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    from starlette.middleware.gzip import GZipMiddleware
    
    # Only add HTTPS redirect if we're behind a reverse proxy
    if os.getenv('EVIDENTLY_BASE_URL', '').startswith('https://'):
        app.add_middleware(HTTPSRedirectMiddleware)
    
    allowed_hosts = ['localhost', '127.0.0.1']
    base_url = os.getenv('EVIDENTLY_BASE_URL', '')
    if base_url:
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        allowed_hosts.append(parsed.hostname)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    app.add_middleware(GZipMiddleware)

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

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Evidently Data Drift API")
    set_drift_state(False, 0.0)  # Reset drift state on startup

# --- API Endpoints ---

@app.get("/health")
async def health():
    """Simple health check endpoint."""
    try:
        # Check if we can load data as a basic health check
        reference_data, current_data = load_data()
        return {"status": "ok", "message": "Data loaded successfully"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        # Return 200 OK with error message instead of 400 Bad Request
        # This prevents Prometheus from marking the service as down
        return {"status": "warning", "message": str(e)}
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
    try:
        content = generate_latest()
        return Response(content=content, media_type="text/plain")
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        # Return empty metrics instead of error to avoid breaking Prometheus
        return Response(content=b"", media_type="text/plain")

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
def reset_drift():
    """
    Resets the drift by copying current data to reference data.
    This will result in a drift score of 0 on the next calculation.
    Also resets any artificial drift state.
    """
    try:
        # Copy current data to reference data to reset drift
        shutil.copy(CURRENT_DATA_PATH, REFERENCE_DATA_PATH)
        logger.info(f"Successfully copied {CURRENT_DATA_PATH} to {REFERENCE_DATA_PATH}")

        # Reset artificial drift state
        set_drift_state(enabled=False, percentage=0.0)
        
        # Reset Prometheus metrics to 0 immediately
        data_drift_score.set(0.0)
        for gauge in feature_drift_scores.values():
            gauge.set(0.0)
            
        logger.info("Drift state has been reset. New reference data is in place.")
        
        return {"status": "Drift reset successfully. Reference data updated."}
    except Exception as e:
        logger.error(f"Error resetting drift: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting drift: {str(e)}")

@app.get("/drift_status")
async def drift_status():
    return get_drift_state()

@app.get("/drift_full_report", response_class=HTMLResponse)
async def drift_full_report():
    """Generate a comprehensive HTML report with all drift information."""
    try:
        # Get drift data
        drift_data = calculate_drift()
        drift_state = get_drift_state()
        
        # Load reference and current data for visualizations
        # Use a sample of the data for faster processing
        reference_data, current_data = load_data()
        
        # Sample data if it's large (more than 1000 rows)
        if len(reference_data) > 1000:
            reference_data = reference_data.sample(1000, random_state=42)
        if len(current_data) > 1000:
            current_data = current_data.sample(1000, random_state=42)
        
        # Generate HTML report
        html_content = generate_drift_report_html(drift_data, drift_state, reference_data, current_data)
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error generating drift report: {e}")
        # Return a simple error page instead of raising an exception
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error - Drift Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                .error-container {{ max-width: 800px; margin: 0 auto; background-color: #fff3f3; padding: 20px; border-radius: 5px; }}
                h1 {{ color: #d32f2f; }}
            </style>
        </head>
        <body>
            <div class="error-container">
                <h1>Error Generating Drift Report</h1>
                <p>There was an error generating the drift report. Please try again later or contact the administrator.</p>
                <p>Error details: {str(e)}</p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=200)  # Return 200 instead of 500 to avoid Streamlit errors

def generate_drift_report_html(drift_data, drift_state, reference_data, current_data):
    """Generate HTML content for the drift report."""
    # Create figures for feature distributions - only for top drifting features
    feature_plots = {}
    
    # Sort features by drift score and take only top 5
    sorted_features = sorted(
        drift_data["feature_drift"].items(),
        key=lambda x: x[1]["drift_score"],
        reverse=True
    )[:5]  # Limit to top 5 features
    
    for feature, data in sorted_features:
        if feature in reference_data.columns and feature in current_data.columns:
            try:
                # Use a smaller figure size and fewer bins for faster rendering
                fig = Figure(figsize=(8, 4))
                ax = fig.subplots()
                
                # Side-by-side histograms so that overlapping distributions restent visibles
                ref_values = reference_data[feature].dropna()
                cur_values = current_data[feature].dropna()
                bins = 15
                # Use the same bin edges for both datasets
                counts_ref, bin_edges = np.histogram(ref_values, bins=bins)
                counts_cur, _ = np.histogram(cur_values, bins=bin_edges)

                bar_width = (bin_edges[1] - bin_edges[0]) * 0.4  # 40 % width
                # Center positions for bars
                centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                ax.bar(centers - bar_width/2, counts_ref, width=bar_width, color="#1f77b4", alpha=0.8, label="Reference")
                ax.bar(centers + bar_width/2, counts_cur, width=bar_width, color="#ff7f0e", alpha=0.8, label="Current")
                
                ax.set_title(f"{feature} Distribution")
                ax.set_xlabel(feature)
                ax.set_ylabel("Count")
                ax.legend()
                
                # Use lower DPI for faster rendering
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=72)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode("utf-8")
                feature_plots[feature] = img_str
            except Exception as e:
                logger.error(f"Error generating plot for {feature}: {e}")
    
    # Generate HTML content
    artificial_drift_html = ""
    if drift_state.get("enabled", False):
        artificial_drift_html = (
            f"""
            <div class="artificial-drift">
                <h3>⚠️ Active Artificial Drift</h3>
                <p>An artificial drift of <strong>{drift_state['percentage']*100}%</strong> is currently applied to the data.</p>
            </div>
            """
        )
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Complete Drift Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ background-color: #4b6584; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .summary {{ background-color: #f5f6fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .feature-card {{ background-color: white; border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 15px; }}
            .feature-header {{ display: flex; justify-content: space-between; align-items: center; }}
            .drift-high {{ color: #e74c3c; font-weight: bold; }}
            .drift-medium {{ color: #f39c12; font-weight: bold; }}
            .drift-low {{ color: #27ae60; font-weight: bold; }}
            .status-indicator {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 5px; }}
            .status-red {{ background-color: #e74c3c; }}
            .status-yellow {{ background-color: #f39c12; }}
            .status-green {{ background-color: #27ae60; }}
            .metric {{ margin-bottom: 10px; }}
            .plot {{ margin-top: 15px; width: 100%; max-width: 800px; }}
            .tabs {{ display: flex; margin-bottom: 20px; }}
            .tab {{ padding: 10px 20px; cursor: pointer; background-color: #f5f6fa; border: 1px solid #ddd; border-bottom: none; }}
            .tab.active {{ background-color: white; border-bottom: 1px solid white; }}
            .tab-content {{ display: none; }}
            .tab-content.active {{ display: block; }}
            .artificial-drift {{ background-color: #fdedec; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #e74c3c; }}
            .timestamp {{ color: #7f8c8d; font-size: 0.9em; margin-top: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Complete Drift Report</h1>
                <p class="timestamp">Generated on {datetime.now().strftime('%d/%m/%Y at %H:%M:%S')}</p>
            </div>
            
            {artificial_drift_html}
            
            <div class="summary">
                <h2>Drift Summary</h2>
                <div class="metric">
                    <strong>Global Drift Score:</strong> 
                    <span class="{"drift-high" if drift_data["drift_score"] > 0.3 else "drift-medium" if drift_data["drift_score"] > 0.1 else "drift-low"}">
                        {drift_data["drift_score"]:.4f}
                    </span>
                </div>
                <div class="metric">
                    <strong>Drift Detected:</strong> 
                    <span class="{"drift-high" if drift_data["drift_detected"] else "drift-low"}">
                        {"Yes" if drift_data["drift_detected"] else "No"}
                    </span>
                </div>
                <div class="metric">
                    <strong>Features with Drift:</strong> {drift_data["drifted_features_count"]} out of {drift_data["total_features_analyzed"]}
                </div>
            </div>
            
            <div class="tabs">
                <div class="tab active" onclick="openTab(event, 'features')">Features</div>
                <div class="tab" onclick="openTab(event, 'data')">Data</div>
            </div>
            
            <div id="features" class="tab-content active">
                <h2>Feature Analysis</h2>
                
    """
    
    # Add feature cards
    for feature, data in drift_data["feature_drift"].items():
        drift_score = data["drift_score"]
        drift_class = "drift-high" if drift_score > 0.3 else "drift-medium" if drift_score > 0.1 else "drift-low"
        status_class = "status-red" if drift_score > 0.3 else "status-yellow" if drift_score > 0.1 else "status-green"
        
        html += f"""
                <div class="feature-card">
                    <div class="feature-header">
                        <h3><span class="status-indicator {status_class}"></span> {feature}</h3>
                        <span class="{drift_class}">Score: {drift_score:.4f}</span>
                    </div>
                    <div class="metric">
                        <strong>p-value:</strong> {data["p_value"]:.6f}
                    </div>
                    <div class="metric">
                        <strong>Drift Detected:</strong> {"Yes" if data["drift_detected"] else "No"}
                    </div>
        """
        
        if feature in feature_plots:
            html += f"""
                    <div class="plot">
                        <img src="data:image/png;base64,{feature_plots[feature]}" alt="{feature} Distribution" style="width:100%">
                    </div>
            """
            
        html += "</div>\n"
    
    html += f"""
            </div>
            
            <div id="data" class="tab-content">
                <h2>Data Overview</h2>
                <h3>Reference Data</h3>
                <div style="overflow-x:auto;">
                    {reference_data.head(10).to_html(classes="table table-striped", border=0)}
                </div>
                
                <h3>Current Data</h3>
                <div style="overflow-x:auto;">
                    {current_data.head(10).to_html(classes="table table-striped", border=0)}
                </div>
            </div>
        </div>
        
        <script>
            function openTab(evt, tabName) {{
                var i, tabcontent, tablinks;
                
                // Hide all tab content
                tabcontent = document.getElementsByClassName("tab-content");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].className = tabcontent[i].className.replace(" active", "");
                }}
                
                // Remove active class from all tabs
                tablinks = document.getElementsByClassName("tab");
                for (i = 0; i < tablinks.length; i++) {{
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }}
                
                // Show the selected tab content and add active class to the button
                document.getElementById(tabName).className += " active";
                evt.currentTarget.className += " active";
            }}
        </script>
    </body>
    </html>
    """
    
    return html

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
