from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse
import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
import os
from datetime import datetime

app = FastAPI()

REFERENCE_DATA_DIR = "/app/reference/"
CURRENT_DATA_DIR = "/app/current/"

def _get_latest_file(directory, prefix):
    """Gets the most recent file in a directory with a given prefix."""
    print(f"--- _get_latest_file: Searching in '{directory}' for prefix '{prefix}' ---")
    if not os.path.exists(directory):
        print(f"--- _get_latest_file: ERROR - Directory does not exist: {directory} ---")
        return None
    if not os.path.isdir(directory):
        print(f"--- _get_latest_file: ERROR - Path is not a directory: {directory} ---")
        return None
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".csv")]
    if not files:
        print(f"--- _get_latest_file: No files found with prefix '{prefix}' in '{directory}' ---")
        return None
    latest_file = max(files)
    print(f"--- _get_latest_file: Found latest file: {latest_file} ---")
    return os.path.join(directory, latest_file)

def _calculate_drift_metrics(noise_factor=None):
    """Helper function to calculate drift metrics and return the report dictionary.
    
    Args:
        noise_factor: Optional factor to simulate different levels of data drift (0.0 to 1.0)
    """
    print(f"--- _calculate_drift_metrics: Entering function (noise_factor={noise_factor}) ---")
    
    reference_file = _get_latest_file(REFERENCE_DATA_DIR, "best_model_data") # Fixed prefix
    current_file = _get_latest_file(CURRENT_DATA_DIR, "current_data") # Fixed prefix

    if not reference_file:
        print("--- _calculate_drift_metrics: ERROR - Reference file not found ---")
        raise FileNotFoundError(f"Reference data file not found starting with 'best_model_data_' in {REFERENCE_DATA_DIR}")
    if not current_file:
        print("--- _calculate_drift_metrics: ERROR - Current data file not found ---")
        raise FileNotFoundError(f"Current data file not found starting with 'current_data_' in {CURRENT_DATA_DIR}")

    print(f"--- _calculate_drift_metrics: Reference file: {reference_file} ---")
    print(f"--- _calculate_drift_metrics: Current file: {current_file} ---")

    try:
        reference_data = pd.read_csv(reference_file)
        current_data = pd.read_csv(current_file)
    except Exception as e:
        print(f"--- _calculate_drift_metrics: ERROR while reading CSV files: {e} ---")
        raise

    print(f"--- _calculate_drift_metrics: Reference data dimensions: {reference_data.shape} ---")
    print(f"--- _calculate_drift_metrics: Current data dimensions: {current_data.shape} ---")

    cols_to_drop = ['grav', 'Num_Acc'] # 'grav' is the target, 'Num_Acc' is often an ID
    reference_data_cols_before_drop = set(reference_data.columns)
    current_data_cols_before_drop = set(current_data.columns)

    reference_data = reference_data.drop(columns=[col for col in cols_to_drop if col in reference_data.columns], errors='ignore')
    current_data = current_data.drop(columns=[col for col in cols_to_drop if col in current_data.columns], errors='ignore')
    
    dropped_from_ref = reference_data_cols_before_drop - set(reference_data.columns)
    dropped_from_cur = current_data_cols_before_drop - set(current_data.columns)
    if dropped_from_ref:
        print(f"--- _calculate_drift_metrics: Columns dropped from reference_data: {dropped_from_ref} ---")
    if dropped_from_cur:
        print(f"--- _calculate_drift_metrics: Columns dropped from current_data: {dropped_from_cur} ---")

    print(f"--- _calculate_drift_metrics: Reference data dimensions (after drop): {reference_data.shape} ---")
    print(f"--- _calculate_drift_metrics: Current data dimensions (after drop): {current_data.shape} ---")

    common_cols = sorted(list(set(reference_data.columns) & set(current_data.columns)))
    if not common_cols:
        print("--- _calculate_drift_metrics: ERROR - No common columns found after potential drops. Please check column names and CSV files. ---")
        raise ValueError("No common columns between reference and current datasets after potential drops.")
    
    print(f"--- _calculate_drift_metrics: Common columns used ({len(common_cols)}): {common_cols} ---")
    reference_data = reference_data[common_cols]
    current_data = current_data[common_cols]

    for col in common_cols:
        ref_unique = reference_data[col].nunique()
        cur_unique = current_data[col].nunique()
        ref_nan = reference_data[col].isna().sum()
        cur_nan = current_data[col].isna().sum()
        print(f"--- Debug col '{col}': REF unique={ref_unique}, nan={ref_nan} | CUR unique={cur_unique}, nan={cur_nan} ---")
        if ref_unique == 1 and len(reference_data) > 1:
            print(f"--- WARNING: Column '{col}' in REFERENCE has only one unique value: {reference_data[col].iloc[0] if len(reference_data) > 0 else 'N/A'} ---")
        if cur_unique == 1 and len(current_data) > 1:
            print(f"--- WARNING: Column '{col}' in CURRENT has only one unique value: {current_data[col].iloc[0] if len(current_data) > 0 else 'N/A'} ---")
    
    # For Evidently 0.7.6, we'll use a simpler approach
    # by calculating the distribution difference for each numeric column
    print("--- _calculate_drift_metrics: Calculating distribution differences ---")
    
    drift_count = 0
    drifted_features = []
    total_columns = len(common_cols)
    
    # For each column, calculate distribution difference
    for col in common_cols:
        if reference_data[col].dtype in ['int64', 'float64'] and reference_data[col].nunique() > 1 and current_data[col].nunique() > 1:
            try:
                # Pearson correlation between distributions
                ref_vals = reference_data[col].dropna().values
                cur_vals = current_data[col].dropna().values
                if len(ref_vals) > 1 and len(cur_vals) > 1:
                    corr = np.corrcoef(np.histogram(ref_vals, bins=20, density=True)[0],
                                       np.histogram(cur_vals, bins=20, density=True)[0])[0, 1]
                    if corr < 0.95:
                        drift_count += 1
                        drifted_features.append(col)
            except Exception as e:
                print(f"--- _calculate_drift_metrics: Error in correlation for column {col}: {e} ---")
        else:
            # For non-numeric or low-variance columns, skip drift detection
            continue
    drift_share = drift_count / total_columns if total_columns > 0 else 0.0
    
    # Create result dictionary
    result_dict = {
        "drift_share": drift_count / total_columns if total_columns > 0 else 0, # Original calculation
        "total_columns": total_columns,
        "drifted_columns": drift_count,
        "drifted_feature_list": drifted_features,
        "status": "success"
    }
    
    print(f"--- _calculate_drift_metrics: Calculation complete - {drift_count} out of {total_columns} columns with drift detected ---") # Original print
    return result_dict

@app.get("/drift_score", response_class=PlainTextResponse)
@app.get("/metrics", response_class=PlainTextResponse)
async def get_drift_score_prometheus(noise: float = None):
    print(f"--- get_drift_score_prometheus: Entering function (noise={noise}) ---") 
    try:
        # Get drift calculation results
        result = _calculate_drift_metrics(noise_factor=noise)
        print(f"--- get_drift_score_prometheus: Results obtained: {result} ---")
        
        # Check if calculation was successful
        if result.get('status') != 'success':
            raise ValueError("Error during data drift calculation")
        
        # Get drift score
        drift_share = float(result.get('drift_share', 0.0))
        drifted_columns = int(result.get('drifted_columns', 0))
        total_columns = int(result.get('total_columns', 1))  # Avoid division by zero
        
        print(f"--- get_drift_score_prometheus: Drift detected in {drifted_columns} out of {total_columns} columns (score: {drift_share:.2f}) ---")

        # Format the response in Prometheus format
        response_lines = [
            "# HELP ml_data_drift_score Drift score between reference and current data (share of drifting features)",
            "# TYPE ml_data_drift_score gauge",
            f"ml_data_drift_score {drift_share:.6f}"
        ]
        
        prometheus_metric = "\n".join(response_lines) + "\n"
        
        print(f"--- get_drift_score_prometheus: FINAL metric Prometheus string BEFORE RETURN ---:\n{prometheus_metric}\n--- END OF METRIC --- (Length: {len(prometheus_metric)})")
        return PlainTextResponse(content=prometheus_metric, media_type="text/plain")

    except FileNotFoundError as e:
        print(f"--- get_drift_score_prometheus: EXCEPTION FileNotFoundError: {e} ---")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        print(f"--- get_drift_score_prometheus: EXCEPTION ValueError: {e} ---")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"--- get_drift_score_prometheus: GENERAL EXCEPTION: {e} ---")
        raise HTTPException(status_code=500, detail=f"Error calculating drift: {str(e)}")

@app.get("/drift_dashboard", response_class=HTMLResponse)
async def get_drift_dashboard():
    try:
        # Get drift calculation results
        result = _calculate_drift_metrics()
        
        # Check if calculation was successful
        if result.get('status') != 'success':
            raise ValueError("Error during data drift calculation")
        
        # Get drift metrics
        drift_share = result.get('drift_share', 0.0)
        drifted_columns = result.get('drifted_columns', 0)
        total_columns = result.get('total_columns', 1)
        
        print(f"--- get_drift_dashboard: Drift detected in {drifted_columns} out of {total_columns} columns (score: {drift_share:.2f}) ---")
        
        html_content = f"""
        <html>
            <head>
                <title>Data Drift Score</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f9; color: #333; text-align: center; }}
                    .container {{ background-color: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.1); display: inline-block; }}
                    h1 {{ color: #2c3e50; }}
                    .score {{ font-size: 3em; color: #3498db; margin: 20px 0; font-weight: bold; }}
                    .timestamp {{ font-size: 0.9em; color: #7f8c8d; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Data Drift Score</h1>
                    <p>(Share of Drifting Features)</p>
                    <div class="score">{drift_share:.2f}</div>
                    <p class="timestamp">Last calculated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except FileNotFoundError as e:
        return HTMLResponse(content=f"<html><body><h1>Error</h1><p>Could not calculate drift: {str(e)}</p><p>Please ensure reference and current data files are available.</p></body></html>", status_code=404)
    except ValueError as e:
        return HTMLResponse(content=f"<html><body><h1>Error</h1><p>Could not calculate drift: {str(e)}</p></body></html>", status_code=400)
    except Exception as e:
        return HTMLResponse(content=f"<html><body><h1>Error</h1><p>An unexpected error occurred: {str(e)}</p></body></html>", status_code=500)

import requests
from fastapi import Request, Response
from fastapi.responses import JSONResponse

@app.get("/drift_report", response_class=JSONResponse)
async def get_drift_report():
    """
    Endpoint to retrieve the complete drift report (score, number of drifted columns, total, list of drifted columns)
    """
    try:
        result = _calculate_drift_metrics()
        if result.get('status') != 'success':
            raise ValueError("Error during data drift calculation")
        # On retourne tout le dictionnaire (score, nombre, liste...)
        return JSONResponse(content=result)
    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/trigger_airflow_from_alert")
async def trigger_airflow_from_alert(request: Request):
    # We ignore the Alertmanager payload content
    airflow_url = "http://airflow-webserver:8080/api/v1/dags/road_accidents/dagRuns"
    auth = ("airflow", "airflow")
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(airflow_url, auth=auth, headers=headers, json={})
        if r.status_code in (200, 201):
            return Response(status_code=200)
        else:
            return Response(content=r.text, status_code=r.status_code)
    except Exception as e:
        return Response(content=str(e), status_code=500)

if __name__ == "__main__":
    import uvicorn
    # This is for local testing. For deployment, use a proper ASGI server like Uvicorn/Gunicorn.
    # The Dockerfile in a real setup would specify how to run this.
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True) # Using port 8001 for the drift service
