import sys
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
import pandas as pd
import numpy as np
import os
from datetime import datetime
import tempfile
import json
from typing import Optional, Dict, Any
from functools import lru_cache

app = FastAPI()

REFERENCE_DATA_DIR = "/app/reference/"
CURRENT_DATA_DIR = "/app/current/"

# Global override for the noise factor applied during drift calculation.
# Allows simulating a constant drift visible in Prometheus.
NOISE_OVERRIDE = None  # None = no override, otherwise float (>0)

# Liste des caractéristiques du modèle (mêmes noms que les colonnes du fichier CSV)
MODEL_FEATURES = ["catu", "sexe", "trajet", "catr", "circ", "vosp", "prof", 
                 "plan", "surf", "situ", "lum", "atm", "col"]

# Types des caractéristiques (toutes numériques dans notre cas)
FEATURE_TYPES = {feature: "num" for feature in MODEL_FEATURES}

@app.post("/config/noise")
async def set_noise_override(payload: Dict[str, Any] = Body(...)):
    """
    Configure the noise override factor for drift simulation.
    Send an empty payload to reset to no noise.
    Send {"noise": 0.8} to set noise to 0.8 (high drift).
    """
    global NOISE_OVERRIDE
    
    # Si le payload est vide, réinitialiser le bruit
    if not payload or len(payload) == 0:
        NOISE_OVERRIDE = None
        return JSONResponse({"status": "success", "message": "Noise override reset to None", "noise": None})
    
    # Si le payload contient une clé 'noise', utiliser cette valeur
    if "noise" in payload:
        try:
            noise_value = float(payload["noise"])
            NOISE_OVERRIDE = noise_value
            return JSONResponse({
                "status": "success", 
                "message": f"Noise override set to {noise_value}", 
                "noise": noise_value
            })
        except (ValueError, TypeError) as e:
            return JSONResponse({
                "status": "error", 
                "message": f"Invalid noise value: {str(e)}"
            }, status_code=400)
    
    # Si le payload ne contient pas de clé 'noise'
    return JSONResponse({
        "status": "error", 
        "message": "Missing 'noise' parameter in payload"
    }, status_code=400)

@app.get("/config/noise")
async def get_noise_override():
    """
    Get the current noise override factor.
    """
    return JSONResponse({
        "status": "success", 
        "noise": NOISE_OVERRIDE
    })

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return JSONResponse({"status": "ok"})

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return JSONResponse({
        "name": "Evidently Drift Control API",
        "version": "1.0.0",
        "endpoints": [
            "/config/noise",
            "/health",
            "/"
        ]
    })

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

@lru_cache(maxsize=1)
def _calculate_drift_metrics(noise_factor=None):
    print(f"--- _calculate_drift_metrics: Entering function (noise_factor={noise_factor}) ---")
    
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
        # Charger les données avec le bon séparateur (virgule, pas point-virgule)
        reference_data = pd.read_csv(reference_file, on_bad_lines='warn', engine='python')
        current_data = pd.read_csv(current_file, on_bad_lines='warn', engine='python')
        
        # Afficher les premières lignes pour le débogage
        print("--- Reference data sample: ---")
        print(reference_data.head())
        print("\n--- Current data sample: ---")
        print(current_data.head())
        
    except Exception as e:
        print(f"--- _calculate_drift_metrics: ERROR while reading CSV files: {e} ---")
        print(f"Reference file path: {reference_file}")
        print(f"Current file path: {current_file}")
        print(f"Current working directory: {os.getcwd()}")
        print("Files in reference directory:", os.listdir(os.path.dirname(reference_file)))
        print("Files in current directory:", os.listdir(os.path.dirname(current_file)))
        raise

    print(f"--- _calculate_drift_metrics: Reference data dimensions: {reference_data.shape} ---")
    print(f"--- _calculate_drift_metrics: Current data dimensions: {current_data.shape} ---")

    # Vérifier les colonnes disponibles
    print("\n--- Reference data columns:", reference_data.columns.tolist())
    print("--- Current data columns:", current_data.columns.tolist())
    
    # Vérifier que toutes les colonnes requises sont présentes
    available_features = [col for col in MODEL_FEATURES if col in reference_data.columns and col in current_data.columns]
    missing_features = set(MODEL_FEATURES) - set(available_features)
    
    if missing_features:
        print(f"--- WARNING: Some model features are missing: {missing_features} ---")
    
    if not available_features:
        error_msg = "No model features available in both reference and current datasets. "
        error_msg += f"Available columns - Reference: {reference_data.columns.tolist()}, "
        error_msg += f"Current: {current_data.columns.tolist()}"
        raise ValueError(error_msg)
        
    print(f"--- _calculate_drift_metrics: Using {len(available_features)} model features: {available_features} ---")
    
    # Sélectionner uniquement les colonnes nécessaires
    reference_data = reference_data[available_features].copy()
    current_data = current_data[available_features].copy()
    
    # Convertir les colonnes en numérique si elles ne le sont pas déjà
    for col in available_features:
        reference_data[col] = pd.to_numeric(reference_data[col], errors='coerce')
        current_data[col] = pd.to_numeric(current_data[col], errors='coerce')
        
        # Remplacer les valeurs manquantes par la moyenne de la colonne
        ref_mean = reference_data[col].mean()
        cur_mean = current_data[col].mean()
        reference_data[col].fillna(ref_mean, inplace=True)
        current_data[col].fillna(cur_mean, inplace=True)
    
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
    
    # Ne pas convertir en catégories car les données sont déjà numériques
    # Cette conversion n'est pas nécessaire pour les données normalisées
    pass
    
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
                    
                # Augmenter le seuil de corrélation à 0.8 pour réduire les faux positifs
                if corr < 0.8:  # Anciennement 0.95
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


@app.get("/drift_full_report", response_class=HTMLResponse)
async def get_drift_full_report(noise: float = None):
    """Return a full Evidently Data Drift report as an HTML page."""
    try:
        print("--- get_drift_full_report: Starting HTML report generation ---")
        reference_file = _get_latest_file(REFERENCE_DATA_DIR, "best_model_data")
        current_file = _get_latest_file(CURRENT_DATA_DIR, "current_data")

        if not reference_file or not current_file:
            print(f"--- get_drift_full_report: File not found - ref: {reference_file}, current: {current_file} ---")
            raise FileNotFoundError("Reference or current data file missing")

        print(f"--- get_drift_full_report: Loading data files - ref: {reference_file}, current: {current_file} ---")
        reference_data = pd.read_csv(reference_file)
        current_data = pd.read_csv(current_file)
        
        print(f"--- get_drift_full_report: Data loaded - ref shape: {reference_data.shape}, current shape: {current_data.shape} ---")
        
        # S'assurer que toutes les colonnes nécessaires sont présentes
        missing_ref = set(MODEL_FEATURES) - set(reference_data.columns)
        missing_cur = set(MODEL_FEATURES) - set(current_data.columns)
        missing_cols = missing_ref.union(missing_cur)
        
        if missing_cols:
            print(f"--- WARNING: Some model features are missing: {missing_cols} ---")

        # If no 'noise' parameter is provided, use the global override
        if noise is None:
            noise = NOISE_OVERRIDE
            if noise is not None:
                print(f"--- get_drift_full_report: Using global NOISE_OVERRIDE={noise} ---")
                
        # Force the drift_share to the noise value if specified
        # This ensures the HTML report will display the same value as the API
        force_drift_share = noise if noise is not None else None
        print(f"--- get_drift_full_report: Force drift_share to {force_drift_share} ---")

        # Apply noise to current data if noise parameter is provided
        if noise is not None and noise > 0:
            print(f"--- get_drift_full_report: Applying noise factor {noise} to current data ---")
            
            # Create a copy of current data to avoid modifying the original
            current_data = current_data.copy()
            
            # Select only numeric columns to apply noise
            numeric_cols = current_data.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                # Apply Gaussian noise to numeric columns
                for col in numeric_cols:
                    # Calculate the column's standard deviation
                    std = current_data[col].std()
                    if std > 0:
                        # Generate Gaussian noise with standard deviation proportional to the noise factor
                        # Use a higher factor to ensure detectable drift
                        noise_values = np.random.normal(0, std * noise * 2, size=len(current_data))
                        # Add noise to current data
                        current_data[col] = current_data[col] + noise_values
                        # Also add a systematic shift to ensure drift
                        if noise > 0.5:  # For high noise factors, add a shift
                            shift = std * noise
                            current_data[col] = current_data[col] + shift
                print(f"--- get_drift_full_report: Enhanced noise applied to {len(numeric_cols)} numeric columns ---")
            
            # For categorical columns, we can shuffle more values
            categorical_cols = current_data.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                # Determine number of items to shuffle (proportional to noise factor)
                # Use a higher factor to ensure detectable drift
                n_samples = int(len(current_data) * noise * 2)
                if n_samples > 0:
                    for col in categorical_cols:
                        # Randomly select indices to modify
                        indices = np.random.choice(len(current_data), size=min(n_samples, len(current_data)), replace=False)
                        # Get unique values from the column
                        unique_values = current_data[col].unique()
                        if len(unique_values) > 1:
                            # For each selected index, assign a different random value
                            for idx in indices:
                                current_value = current_data.loc[idx, col]
                                # Select a value different from the current one
                                other_values = [v for v in unique_values if v != current_value]
                                if other_values:
                                    current_data.loc[idx, col] = np.random.choice(other_values)
                            
                            # For high noise factors, also introduce new categories
                            if noise > 0.5 and len(indices) > 0:
                                # Create a new category
                                new_category = f"NEW_VALUE_{col}"
                                # Replace some values with the new category
                                n_new = max(1, int(len(indices) * 0.2))
                                new_indices = np.random.choice(indices, size=n_new, replace=False)
                                current_data.loc[new_indices, col] = new_category
                    
                    print(f"--- get_drift_full_report: Enhanced shuffling in {len(categorical_cols)} categorical columns for {n_samples} samples ---")
        
        # Use our custom function to generate a simple HTML report
        print("--- get_drift_full_report: Generating custom HTML report ---")
        try:
            html = generate_simple_html_report(reference_data, current_data, force_drift_share)
            print(f"--- get_drift_full_report: Custom HTML report generated, length: {len(html)} ---")
        except Exception as e:
            print(f"--- get_drift_full_report: Error generating custom HTML report: {e} ---")
            raise
        
        if not html or len(html) < 100:
            raise ValueError(f"Generated HTML is too short or empty: {len(html)} bytes")
        
        print("--- get_drift_full_report: Successfully generated HTML report ---")
        return HTMLResponse(content=html)
                
    except FileNotFoundError as e:
        print(f"--- get_drift_full_report: FileNotFoundError: {e} ---")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"--- get_drift_full_report: Exception: {e} ---")
        raise HTTPException(status_code=500, detail=f"Unable to generate report: {e}")



def generate_simple_html_report(reference_data, current_data, force_drift_share=None):
    """
    Generate a simple HTML report comparing reference and current data distributions.
    
    Args:
        reference_data (pd.DataFrame): Reference dataset
        current_data (pd.DataFrame): Current dataset to compare against reference
        force_drift_share (float, optional): If provided, override the calculated drift share with this value
        
    Returns:
        str: HTML report as string
    """
    import matplotlib.pyplot as plt
    import io
    import base64
    from scipy import stats
    import numpy as np
    
    # Ensure we have the same columns in both datasets
    common_columns = set(reference_data.columns).intersection(set(current_data.columns))
    common_columns = [col for col in common_columns if col in MODEL_FEATURES]
    
    if not common_columns:
        return "<h1>Error: No common columns found between reference and current data</h1>"
    
    # Calculate drift metrics
    drift_count = 0
    drifted_features = []
    total_columns = len(common_columns)
    
    for col in common_columns:
        try:
            ref_vals = reference_data[col].dropna()
            cur_vals = current_data[col].dropna()
            
            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue
                
            # Determine if column is numeric or categorical
            if pd.api.types.is_numeric_dtype(reference_data[col]):
                # Numeric column - use correlation
                try:
                    # Create histograms for reference and current data
                    ref_hist, ref_bins = np.histogram(ref_vals, bins=20)
                    cur_hist, _ = np.histogram(cur_vals, bins=ref_bins)
                    
                    # Normalize histograms
                    if ref_hist.sum() > 0:
                        ref_hist = ref_hist / ref_hist.sum()
                    if cur_hist.sum() > 0:
                        cur_hist = cur_hist / cur_hist.sum()
                    
                    # Calculate correlation between histograms
                    if np.std(ref_hist) > 0 and np.std(cur_hist) > 0:
                        corr = np.corrcoef(ref_hist, cur_hist)[0, 1]
                        
                        # If correlation is low, consider it drift
                        if corr < 0.9:
                            drift_count += 1
                            drifted_features.append(f"{col} (corr: {corr:.2f})")
                except Exception as e:
                    print(f"Error calculating drift for numeric column {col}: {e}")
            else:
                # Categorical column - use Total Variation Distance (TVD)
                ref_counts = ref_vals.value_counts(normalize=True)
                cur_counts = cur_vals.value_counts(normalize=True)
                
                # Get all unique categories
                all_cats = set(ref_counts.index).union(set(cur_counts.index))
                
                # Calculate TVD
                tvd = 0
                for cat in all_cats:
                    ref_prob = ref_counts.get(cat, 0)
                    cur_prob = cur_counts.get(cat, 0)
                    tvd += abs(ref_prob - cur_prob)
                
                tvd = tvd / 2  # Normalize TVD to [0, 1]
                
                # If TVD is high, consider it drift
                if tvd > 0.2:
                    drift_count += 1
                    drifted_features.append(f"{col} (tvd: {tvd:.2f})")
        except Exception as e:
            print(f"Error processing column {col}: {e}")
    
    # Calculate drift share
    drift_share = drift_count / total_columns if total_columns > 0 else 0.0
    
    # Override drift share if requested
    if force_drift_share is not None:
        drift_share = force_drift_share
    
    # Generate plots for each column
    plots_html = []
    for col in common_columns[:10]:  # Limit to first 10 columns to avoid huge reports
        try:
            plt.figure(figsize=(10, 6))
            
            if pd.api.types.is_numeric_dtype(reference_data[col]):
                # Numeric column - plot histograms
                plt.hist(reference_data[col].dropna(), bins=20, alpha=0.5, label='Reference')
                plt.hist(current_data[col].dropna(), bins=20, alpha=0.5, label='Current')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.legend()
                plt.title(f'Distribution of {col}')
            else:
                # Categorical column - plot bar charts
                ref_counts = reference_data[col].value_counts(normalize=True)
                cur_counts = current_data[col].value_counts(normalize=True)
                
                # Get top categories
                top_cats = set(ref_counts.nlargest(10).index).union(set(cur_counts.nlargest(10).index))
                top_cats = list(top_cats)[:10]  # Limit to 10 categories
                
                x = np.arange(len(top_cats))
                width = 0.35
                
                plt.bar(x - width/2, [ref_counts.get(cat, 0) for cat in top_cats], width, label='Reference')
                plt.bar(x + width/2, [cur_counts.get(cat, 0) for cat in top_cats], width, label='Current')
                plt.xlabel('Category')
                plt.ylabel('Frequency')
                plt.xticks(x, top_cats, rotation=45)
                plt.legend()
                plt.title(f'Distribution of {col}')
                plt.tight_layout()
            
            # Convert plot to base64 image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            plots_html.append(f'''
            <div class="plot-container">
                <h3>{col}</h3>
                <img src="data:image/png;base64,{img_data}" alt="Distribution of {col}">
            </div>
            ''')
        except Exception as e:
            plots_html.append(f'''
            <div class="plot-container">
                <h3>{col}</h3>
                <p class="error">Error generating plot: {str(e)}</p>
            </div>
            ''')
    
    # Create HTML report
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Drift Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .drift-score {{ font-size: 24px; font-weight: bold; }}
            .drift-high {{ color: #dc3545; }}
            .drift-medium {{ color: #fd7e14; }}
            .drift-low {{ color: #28a745; }}
            .plot-container {{ margin-bottom: 30px; background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            .controls {{ margin-bottom: 20px; padding: 15px; background-color: #e9ecef; border-radius: 5px; }}
            button {{ padding: 10px 15px; margin-right: 10px; border: none; border-radius: 4px; cursor: pointer; }}
            .btn-primary {{ background-color: #007bff; color: white; }}
            .btn-warning {{ background-color: #ffc107; color: black; }}
            .btn-danger {{ background-color: #dc3545; color: white; }}
            .error {{ color: #dc3545; }}
        </style>
        <script>
            function setNoise(value) {{
                fetch('/config/noise', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ noise: value }})
                }})
                .then(response => response.json())
                .then(data => {{
                    if (data.status === 'success') {{
                        alert('Noise factor set to ' + value + '. Refresh the page to see changes.');
                    }} else {{
                        alert('Error: ' + data.message);
                    }}
                }})
                .catch((error) => {{
                    alert('Error: ' + error);
                }});
            }}
            
            function resetNoise() {{
                fetch('/config/noise', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{}})  // Empty payload resets noise
                }})
                .then(response => response.json())
                .then(data => {{
                    if (data.status === 'success') {{
                        alert('Noise factor reset. Refresh the page to see changes.');
                    }} else {{
                        alert('Error: ' + data.message);
                    }}
                }})
                .catch((error) => {{
                    alert('Error: ' + error);
                }});
            }}
            
            function checkApiStatus() {{
                fetch('/health')
                .then(response => {{
                    if (response.ok) {{
                        document.getElementById('api-status').textContent = 'API is accessible';
                        document.getElementById('api-status').className = 'drift-low';
                        document.getElementById('controls').style.display = 'block';
                        document.getElementById('api-error').style.display = 'none';
                    }} else {{
                        throw new Error('API health check failed');
                    }}
                }})
                .catch((error) => {{
                    document.getElementById('api-status').textContent = 'API is not accessible';
                    document.getElementById('api-status').className = 'drift-high';
                    document.getElementById('controls').style.display = 'none';
                    document.getElementById('api-error').style.display = 'block';
                }});
            }}
            
            window.onload = function() {{
                checkApiStatus();
            }};
        </script>
    </head>
    <body>
        <div class="header">
            <h1>Data Drift Report</h1>
            <p>Comparison between reference and current data distributions</p>
            <p>Reference data: {reference_data.shape[0]} rows, Current data: {current_data.shape[0]} rows</p>
            <p>Drift score: <span class="drift-score {{'drift-high' if drift_share > 0.5 else 'drift-medium' if drift_share > 0.2 else 'drift-low'}}">{drift_share:.2f}</span></p>
            <p id="api-status">Checking API status...</p>
        </div>
        
        <div id="api-error" style="display: none; padding: 15px; background-color: #f8d7da; color: #721c24; border-radius: 5px; margin-bottom: 20px;">
            <strong>L'API de contrôle de drift n'est pas accessible. Les boutons sont désactivés.</strong>
        </div>
        
        <div id="controls" class="controls">
            <h2>Contrôle du Drift</h2>
            <p>Utilisez ces boutons pour simuler différents niveaux de drift dans les données:</p>
            <button class="btn-primary" onclick="resetNoise()">Réinitialiser (No Drift)</button>
            <button class="btn-primary" onclick="setNoise(0.2)">Faible Drift (0.2)</button>
            <button class="btn-warning" onclick="setNoise(0.5)">Moyen Drift (0.5)</button>
            <button class="btn-danger" onclick="setNoise(0.8)">Fort Drift (0.8)</button>
        </div>
        
        <h2>Drifted Features ({len(drifted_features)})</h2>
        <ul>
            {''.join([f'<li>{feature}</li>' for feature in drifted_features])}
        </ul>
        
        <h2>Feature Distributions</h2>
        {''.join(plots_html)}
    </body>
    </html>
    '''
    
    return html

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
