import sys
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

# Global override for the noise factor applied during drift calculation.
# Allows simulating a constant drift visible in Prometheus.
NOISE_OVERRIDE = None  # None = no override, otherwise float (>0)

def _get_latest_file(directory, prefix):
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
    print(f"--- _calculate_drift_metrics: Entering function (noise_factor={noise_factor}) ---")
    
    MODEL_FEATURES = ["catu", "sexe", "trajet", "catr", "circ", "vosp", "prof", 
                     "plan", "surf", "situ", "lum", "atm", "col"]
    
    reference_file = _get_latest_file(REFERENCE_DATA_DIR, "best_model_data") 
    current_file = _get_latest_file(CURRENT_DATA_DIR, "current_data") 

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

    available_features = [col for col in MODEL_FEATURES if col in reference_data.columns and col in current_data.columns]
    missing_features = set(MODEL_FEATURES) - set(available_features)
    
    if missing_features:
        print(f"--- WARNING: Some model features are missing: {missing_features} ---")
    
    if not available_features:
        raise ValueError("No model features available in both reference and current datasets")
        
    print(f"--- _calculate_drift_metrics: Using {len(available_features)} model features: {available_features} ---")
    
    reference_data = reference_data[available_features].copy()
    current_data = current_data[available_features].copy()
    
    # ADD NOISE IF noise_factor PARAMETER IS SPECIFIED
    if noise_factor is not None and noise_factor > 0:
        print(f"--- _calculate_drift_metrics: Applying noise factor {noise_factor} to current data ---")
        
        # Ajouter du bruit aux données courantes pour simuler le drift
        for col in available_features:
            if current_data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # For numerical columns, add Gaussian noise
                if current_data[col].std() > 0:
                    noise = np.random.normal(0, noise_factor * current_data[col].std(), len(current_data))
                    current_data[col] = current_data[col] + noise
                    print(f"--- _calculate_drift_metrics: Added numerical noise to column '{col}' ---")
            else:
                # For categorical columns, randomly shuffle a percentage of values
                if noise_factor > 0.1:  # Si le bruit est significatif
                    n_to_shuffle = int(len(current_data) * min(noise_factor, 1.0))
                    if n_to_shuffle > 0:
                        indices_to_shuffle = np.random.choice(len(current_data), n_to_shuffle, replace=False)
                        shuffled_values = current_data[col].iloc[indices_to_shuffle].sample(frac=1).values
                        current_data.loc[current_data.index[indices_to_shuffle], col] = shuffled_values
                        print(f"--- _calculate_drift_metrics: Added categorical noise to column '{col}' (shuffled {n_to_shuffle} values) ---")
        
        print(f"--- _calculate_drift_metrics: Noise applied successfully ---")
    
    for col in available_features:
        if reference_data[col].dtype == 'object' or reference_data[col].nunique() < 20:
            reference_data[col] = reference_data[col].astype('category').cat.codes
            current_data[col] = current_data[col].astype('category').cat.codes
    
    common_cols = available_features

    for col in common_cols:
        ref_unique = reference_data[col].nunique()
        cur_unique = current_data[col].nunique()
        ref_nan = reference_data[col].isna().sum()
        cur_nan = current_data[col].isna().sum()
        if ref_unique == 1 and len(reference_data) > 1:
            print(f"--- WARNING: Column '{col}' in REFERENCE has only one unique value: {reference_data[col].iloc[0] if len(reference_data) > 0 else 'N/A'} ---")
        if cur_unique == 1 and len(current_data) > 1:
            print(f"--- WARNING: Column '{col}' in CURRENT has only one unique value: {current_data[col].iloc[0] if len(current_data) > 0 else 'N/A'} ---")
    
    print("--- _calculate_drift_metrics: Calculating distribution differences ---")
    
    drift_count = 0
    drifted_features = []
    total_columns = len(common_cols)
    
    for col in common_cols:
        try:
            ref_vals = reference_data[col].dropna().values
            cur_vals = current_data[col].dropna().values
            
            if len(ref_vals) <= 1 or len(cur_vals) <= 1:
                print(f"--- _calculate_drift_metrics: Skipping column '{col}' - not enough data (ref: {len(ref_vals)}, cur: {len(cur_vals)})")
                continue
                
            if np.issubdtype(reference_data[col].dtype, np.number) and reference_data[col].nunique() > 1:
                ref_hist = np.histogram(ref_vals, bins=20, density=True)[0]
                cur_hist = np.histogram(cur_vals, bins=20, density=True)[0]
                
                if np.all(ref_hist == ref_hist[0]) or np.all(cur_hist == cur_hist[0]):
                    print(f"--- _calculate_drift_metrics: Skipping numeric column '{col}' - constant histogram")
                    continue
                    
                corr = np.corrcoef(ref_hist, cur_hist)[0, 1]
                if np.isnan(corr):
                    print(f"--- _calculate_drift_metrics: NaN correlation for column '{col}'", file=sys.stderr)
                    continue
                    
                if corr < 0.95:
                    drift_count += 1
                    drifted_features.append(f"{col} (corr: {corr:.2f})")
                    print(f"--- _calculate_drift_metrics: Drift detected in column '{col}' (corr: {corr:.2f})")
                else:
                    print(f"--- _calculate_drift_metrics: No drift in column '{col}' (corr: {corr:.2f})")
            else:
                ref_counts = reference_data[col].value_counts(normalize=True)
                cur_counts = current_data[col].value_counts(normalize=True)
                
                all_cats = set(ref_counts.index).union(set(cur_counts.index))
                
                tvd = 0.5 * sum(abs(ref_counts.get(cat, 0) - cur_counts.get(cat, 0)) for cat in all_cats)
                
                if tvd > 0.1:
                    drift_count += 1
                    drifted_features.append(f"{col} (tvd: {tvd:.2f})")
                    print(f"--- _calculate_drift_metrics: Categorical drift in column '{col}' (tvd: {tvd:.2f})")
                else:
                    print(f"--- _calculate_drift_metrics: No categorical drift in column '{col}' (tvd: {tvd:.2f})")
                    
        except Exception as e:
            print(f"--- _calculate_drift_metrics: Error processing column '{col}': {str(e)}", file=sys.stderr)
            continue
    
    drift_share = drift_count / total_columns if total_columns > 0 else 0.0
    
    # IF NOISE IS APPLIED, FORCE A HIGH DRIFT FOR TESTING
    if noise_factor is not None and noise_factor > 0:
        # Force a high drift proportional to noise_factor
        forced_drift_share = min(noise_factor, 1.0)  # Maximum 1.0
        print(f"--- _calculate_drift_metrics: FORCING drift_share to {forced_drift_share} due to noise_factor={noise_factor} ---")
        result_dict = {
            "drift_share": forced_drift_share,
            "total_columns": total_columns,
            "drifted_columns": int(forced_drift_share * total_columns),
            "drifted_feature_list": drifted_features + [f"SIMULATED_DRIFT (noise: {noise_factor})"],
            "status": "success"
        }
    else:
        result_dict = {
            "drift_share": drift_share,
            "total_columns": total_columns,
            "drifted_columns": drift_count,
            "drifted_feature_list": drifted_features,
            "status": "success"
        }
    
    print(f"--- _calculate_drift_metrics: Calculation complete - {drift_count} out of {total_columns} columns with drift detected ---")
    return result_dict

@app.get("/drift_score", response_class=PlainTextResponse)
@app.get("/metrics", response_class=PlainTextResponse)
async def get_drift_score_prometheus(noise: float = None):
    print(f"--- get_drift_score_prometheus: Entering function (noise={noise}) ---") 
    # If no noise parameter in the request and an override is defined, apply it
    if noise is None and NOISE_OVERRIDE is not None:
        noise = NOISE_OVERRIDE
    try:
        # Get drift calculation results
        result = _calculate_drift_metrics(noise_factor=noise)
        print(f"--- get_drift_score_prometheus: Results obtained: {result} ---")
        
        # Check if calculation was successful
        if result.get('status') != 'success':
            raise ValueError("Error during data drift calculation")
        
        drift_share = float(result.get('drift_share', 0.0))
        drifted_columns = int(result.get('drifted_columns', 0))
        total_columns = int(result.get('total_columns', 1))
        
        print(f"--- get_drift_score_prometheus: Drift detected in {drifted_columns} out of {total_columns} columns (score: {drift_share:.2f}) ---")

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
        result = _calculate_drift_metrics()
        
        if result.get('status') != 'success':
            raise ValueError("Error during data drift calculation")
        
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
    try:
        result = _calculate_drift_metrics()
        if result.get('status') != 'success':
            raise ValueError("Error during data drift calculation")
        return JSONResponse(content=result)
    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/trigger_airflow_from_alert")
async def trigger_airflow_from_alert(request: Request):
    
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

@app.post("/config/noise")
async def set_noise(config: dict):
    """Set or reset the global noise factor.
    Example call:
        curl -X POST http://localhost:8001/config/noise -H "Content-Type: application/json" -d '{"noise": 0.8}'
    Send `{}` or `{"noise": null}` to reset.
    """
    global NOISE_OVERRIDE
    noise_val = config.get("noise") if isinstance(config, dict) else None
    if noise_val is None:
        NOISE_OVERRIDE = None
        return {"status": "success", "message": "Noise override cleared"}
    try:
        NOISE_OVERRIDE = float(noise_val)
        if NOISE_OVERRIDE < 0:
            raise ValueError
        return {"status": "success", "noise": NOISE_OVERRIDE}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid noise value; must be a positive number or null.")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
