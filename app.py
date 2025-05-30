import streamlit as st
import pandas as pd
import requests
import psycopg2
from sqlalchemy import create_engine
import os
import time
import json # For sending data to API
import plotly.express as px

# --- Configuration ---
# These should ideally be configurable, e.g., via environment variables or a config section in Streamlit
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "road_accidents")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
API_ENDPOINT_PREDICT = os.getenv("API_ENDPOINT_PREDICT", "http://localhost:8000/protected/predict")
API_ENDPOINT_TOKEN = os.getenv("API_ENDPOINT_TOKEN", "http://localhost:8000/token")

# --- Session State Initialization ---
if 'token' not in st.session_state:
    st.session_state.token = None
if 'username' not in st.session_state:
    st.session_state.username = None

# --- Helper Functions ---
def check_pipeline_step_status(marker_file_path):
    """Checks if a marker file exists, indicating a pipeline step is complete."""
    if os.path.exists(marker_file_path):
        return "✅ Terminé"
    return "⏳ En attente / En cours"

def get_year_from_config():
    """
    Tries to get the year from common config or defaults.
    For now, defaults to 2023 as it's a common year in the project.
    A more robust way would be to parse 'config.yaml' used by train_model.py.
    """
    # Check for year-specific marker files if they exist
    for year_candidate in ["2023", "2022", "2021"]: # Add other relevant years
        if os.path.exists(f"data/interim/accidents_{year_candidate}_synthet.done"):
            return year_candidate
    return "2023" # Default year

YEAR = get_year_from_config()

MARKER_FILES = {
    "Extraction des données": "data/raw/extract_data.done",
    "Synthétisation des données": f"data/interim/accidents_{YEAR}_synthet.done",
    "Préparation des données": "data/processed/prepared_data.done",
    "Entraînement du modèle": "models/train_model.done",
    # "Importation des données/métriques": "import_data.done" # Add if import_data.py creates its own marker
}

FEATURE_DEFINITIONS = {
    "catu": {
        "label": "Catégorie d'usager", 
        "options": [1, 2, 3, 4], 
        "option_labels": {1: "Conducteur", 2: "Passager", 3: "Piéton", 4: "Autre (roller, trottinette...)"},
        "default": 1, 
        "type": "selectbox", 
        "help": "1: Conducteur, 2: Passager, 3: Piéton"
    },
    "sexe": {
        "label": "Sexe", 
        "options": [1, 2], 
        "option_labels": {1: "Masculin", 2: "Féminin"},
        "default": 1, 
        "type": "selectbox", 
        "help": "1: Masculin, 2: Féminin"
    },
    "trajet": {
        "label": "Motif du déplacement", 
        "options": [0, 1, 2, 3, 4, 5, 9], 
        "option_labels": {0: "Non renseigné", 1: "Domicile-travail", 2: "Domicile-école", 3: "Courses/Achats", 4: "Utilisation professionnelle", 5: "Promenade/Loisirs", 9: "Autre"},
        "default": 1, 
        "type": "selectbox", 
        "help": "0: Non renseigné, 1: Domicile-travail, 2: Domicile-école, 3: Courses, 4: Professionnel, 5: Loisirs, 9: Autre"
    },
    "catr": {
        "label": "Catégorie de route", 
        "options": [1, 2, 3, 4, 5, 6, 7, 9], 
        "option_labels": {1: "Autoroute", 2: "Route nationale", 3: "Route Départementale", 4: "Voie Communale", 5: "Hors réseau public", 6: "Parc de stationnement ouvert à la circulation publique", 7: "Routes de métropole d'outre-mer", 9: "Autre"},
        "default": 3, 
        "type": "selectbox", 
        "help": "e.g., 1: Autoroute, 3: Route Départementale"
    },
    "circ": {
        "label": "Régime de circulation", 
        "options": [-1, 1, 2, 3, 4], 
        "option_labels": {-1: "Non renseigné", 1: "A sens unique", 2: "Bidirectionnelle", 3: "A chaussées séparées", 4: "Avec voies d'affectation variable"},
        "default": 2, 
        "type": "selectbox", 
        "help": "e.g., 1: A sens unique, 2: Bidirectionnelle"
    },
    "vosp": {
        "label": "Voie réservée", 
        "options": [-1, 0, 1, 2, 3], 
        "option_labels": {-1: "Non renseigné", 0: "Sans objet", 1: "Piste cyclable", 2: "Voie réservée véhicules lents", 3: "Voie réservée bus"},
        "default": 0, 
        "type": "selectbox", 
        "help": "0: Sans objet, 1: Piste cyclable"
    },
    "prof": {
        "label": "Profil en long (pente)", 
        "options": [-1, 1, 2, 3, 4], 
        "option_labels": {-1: "Non renseigné", 1: "Plat", 2: "Pente", 3: "Sommet de côte", 4: "Bas de côte"},
        "default": 1, 
        "type": "selectbox", 
        "help": "1: Plat, 2: Pente"
    },
    "plan": {
        "label": "Tracé en plan (virage)", 
        "options": [-1, 1, 2, 3, 4], 
        "option_labels": {-1: "Non renseigné", 1: "Partie rectiligne", 2: "En courbe à gauche", 3: "En courbe à droite", 4: 'En "S"'},
        "default": 1, 
        "type": "selectbox", 
        "help": "1: Rectiligne, 2: Courbe à gauche"
    },
    "surf": {
        "label": "État de la surface", 
        "options": [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        "option_labels": {-1: "Non renseigné", 1: "Normale", 2: "Mouillée", 3: "Flaques", 4: "Inondée", 5: "Enneigée", 6: "Boue", 7: "Verglacée", 8: "Corps gras - huile", 9: "Autre"},
        "default": 1, 
        "type": "selectbox", 
        "help": "1: Normale, 2: Mouillée, 3: Verglacée"
    },
    "situ": {
        "label": "Situation de l'accident", 
        "options": [-1, 0, 1, 2, 3, 4, 5, 6, 8], 
        "option_labels": {-1: "Non renseigné", 0: "Aucun", 1: "Sur chaussée", 2: "Sur bande d'arrêt d'urgence", 3: "Sur accotement", 4: "Sur trottoir", 5: "Sur piste cyclable", 6: "Sur autre voie spéciale", 8: "Intersection"},
        "default": 1, 
        "type": "selectbox", 
        "help": "1: Sur chaussée, 2: Sur bande d'arrêt d'urgence"
    },
    "lum": {
        "label": "Conditions d'éclairage", 
        "options": [1, 2, 3, 4, 5], 
        "option_labels": {1: "Plein jour", 2: "Crépuscule ou aube", 3: "Nuit sans éclairage public", 4: "Nuit avec éclairage public non allumé", 5: "Nuit avec éclairage public allumé"},
        "default": 1, 
        "type": "selectbox", 
        "help": "1: Plein jour, 2: Crépuscule/aube, 3: Nuit sans éclairage public"
    },
    "atm": {
        "label": "Conditions atmosphériques", 
        "options": [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        "option_labels": {-1: "Non renseigné", 1: "Normale", 2: "Pluie légère", 3: "Pluie forte", 4: "Neige - grêle", 5: "Brouillard - fumée", 6: "Vent fort - tempête", 7: "Temps éblouissant", 8: "Temps couvert", 9: "Autre"},
        "default": 1, 
        "type": "selectbox", 
        "help": "1: Normale, 2: Pluie légère, 8: Brouillard"
    },
    "col": {
        "label": "Type de collision", 
        "options": [-1, 1, 2, 3, 4, 5, 6, 7], 
        "option_labels": {-1: "Non renseigné", 1: "Deux véhicules - frontale", 2: "Deux véhicules - par l'arrière", 3: "Deux véhicules - par le coté", 4: "Trois véhicules et plus - en chaîne", 5: "Trois véhicules et plus - collisions multiples", 6: "Autre collision", 7: "Sans collision"},
        "default": 1, 
        "type": "selectbox", 
        "help": "1: Deux véhicules - frontale, 2: Deux véhicules - par l'arrière"
    }
}
GRAV_MAPPING = {1: "Indemne", 2: "Blessé léger", 3: "Blessé hospitalisé", 4: "Tué"}
MODEL_PREDICTION_LABEL_MAPPING = {
    0: "Pas Grave",  # Sortie du modèle pour un accident non grave
    1: "Grave"       # Sortie du modèle pour un accident grave
}

def fetch_accidents_data():
    """Fetches all data from the 'accidents' table using SQLAlchemy."""
    try:
        # Construct the database URL for SQLAlchemy: postgresql://user:password@host:port/dbname
        db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(db_url)
        query = "SELECT * FROM accidents;"
        # Pandas read_sql_query with an SQLAlchemy engine handles connection management
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as error: # Catch a broader exception for SQLAlchemy related issues
        st.error(f"Erreur lors de la récupération des données des accidents avec SQLAlchemy : {error}")
        return pd.DataFrame()


def fetch_best_model_metrics():
    conn = None
    try:
        conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        cur = conn.cursor()
        # Select individual metric columns and order by run_date
        cur.execute("""
            SELECT 
                run_id, run_date, model_name, accuracy, 
                precision_macro_avg, recall_macro_avg, f1_macro_avg, 
                model_version, model_stage, year 
            FROM best_model_metrics 
            ORDER BY run_date DESC LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            colnames = [desc[0] for desc in cur.description]
            metrics_dict = dict(zip(colnames, row))
            # Convert run_date to string if it's a datetime object, as JSON would have done
            if 'run_date' in metrics_dict and hasattr(metrics_dict['run_date'], 'isoformat'):
                metrics_dict['run_date'] = metrics_dict['run_date'].isoformat()
            return metrics_dict
        return None
    except psycopg2.Error as e:
        st.error(f"Erreur de connexion à la base de données : {e}")
        return None
    finally:
        if conn: conn.close()

def login_user(username, password):
    """Attempts to log in the user by fetching a token from the API."""
    try:
        response = requests.post(API_ENDPOINT_TOKEN, data={"username": username, "password": password}, timeout=10)
        response.raise_for_status()
        token_data = response.json()
        st.session_state.token = token_data["access_token"]
        st.session_state.username = username
        st.success(f"Connecté en tant que {username}!")
        return True
    except requests.exceptions.HTTPError as errh:
        if errh.response.status_code == 401:
            st.error("Nom d'utilisateur ou mot de passe incorrect.")
        else:
            st.error(f"Erreur d'authentification API : {errh}")
            if errh.response is not None:
                st.caption(f"Détail : {errh.response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion lors de l'authentification : {e}")
    return False

def logout_user():
    """Logs out the user by clearing the token from session state."""
    st.session_state.token = None
    st.session_state.username = None
    st.info("Vous avez été déconnecté.")

def predict_live(feature_inputs):
    """Sends feature data to the API for live prediction."""
    if not st.session_state.token:
        st.error("Vous devez être connecté pour effectuer une prédiction.")
        return None

    payload = {
        "features": [feature_inputs] # L'API attend une liste de dictionnaires de features
    }
    # Convertir le payload en DataFrame puis en CSV
    df_payload = pd.DataFrame(feature_inputs, index=[0]) # Crée un DataFrame avec une seule ligne
    csv_payload = df_payload.to_csv(index=False)

    files = {'file_request': ('live_prediction.csv', csv_payload, 'text/csv')}
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    try:
        response = requests.post(API_ENDPOINT_PREDICT, files=files, headers=headers, timeout=10) # timeout de 10s
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur API de prédiction : {e}")
        return None
    except json.JSONDecodeError:
        st.error("Erreur décodage JSON de l'API.")
        return None

st.set_page_config(page_title="Démo MLOps Accidents", layout="wide")
st.title("Démo MLOps - Analyse des Accidents de la Route 2023")
st.markdown("Bienvenue sur l'application de démonstration du projet MLOps.")

# --- Authentication UI ---
if st.session_state.token is None:
    st.sidebar.subheader("Connexion")
    with st.sidebar.form("login_form"):
        username = st.text_input("Nom d'utilisateur", value="johndoe") # Default to johndoe for convenience
        password = st.text_input("Mot de passe", type="password", value="johnsecret") # Default password
        login_button = st.form_submit_button("Se connecter")
        if login_button:
            login_user(username, password)
else:
    st.sidebar.subheader(f"Connecté en tant que: {st.session_state.username}")
    if st.sidebar.button("Se déconnecter"):
        logout_user()
        st.rerun() # Remplacer experimental_rerun par rerun

# Le reste de l'application ne s'affiche que si l'utilisateur est connecté
if st.session_state.token:
    st.header("Liste des Accidents")
    accidents_df = fetch_accidents_data()
    if not accidents_df.empty:
        st.dataframe(accidents_df)

        # Ajouter un diagramme circulaire pour la gravité des accidents
        if 'grav' in accidents_df.columns:
            gravity_counts = accidents_df['grav'].value_counts().reset_index()
            gravity_counts.columns = ['grav', 'count']
            fig = px.pie(gravity_counts, values='count', names='grav', 
                         title='Répartition des Accidents par Gravité',
                         color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("La colonne 'grav' est introuvable pour générer le graphique de gravité.")
    else:
        st.warning("Aucune donnée d'accident trouvée ou erreur lors du chargement.")

    st.header("Performance du Meilleur Modèle")
    metrics = fetch_best_model_metrics()
    if metrics:
        st.markdown("Métriques du modèle 'best_model' (depuis PostgreSQL via MLflow):")
        col1, col2 = st.columns(2)
        if "accuracy" in metrics: col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        
        f1_key = next((k for k in ["f1_macro_avg", "f1-score_weighted", "weighted avg_f1-score"] if k in metrics), None)
        if f1_key: col2.metric("F1-Score (Weighted)", f"{metrics[f1_key]:.4f}")

        cls_report_key_dict = next((k for k in ["classification_report_dict", "classification_report"] if k in metrics and isinstance(metrics[k], dict)), None)
        cls_report_key_str = next((k for k in ["classification_report_str", "classification_report"] if k in metrics and isinstance(metrics[k], str)), None)

        if cls_report_key_dict:
            st.subheader("Rapport de Classification")
            try: st.dataframe(pd.DataFrame(metrics[cls_report_key_dict]).transpose())
            except Exception: st.json(metrics[cls_report_key_dict])
        elif cls_report_key_str:
            st.subheader("Rapport de Classification")
            st.text(metrics[cls_report_key_str])
        elif metrics: 
            st.info("Rapport de classification détaillé non trouvé. Affichage des autres métriques disponibles :")
            
            metrics_for_table = {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
            remaining_complex_metrics = {k: v for k, v in metrics.items() if isinstance(v, (dict, list))}

            if metrics_for_table:
                try:
                    # Convertir toutes les valeurs en chaînes de caractères pour la colonne 'Valeur'
                    items_for_df = [(str(k), str(v)) for k, v in metrics_for_table.items()]
                    metrics_df = pd.DataFrame(items_for_df, columns=['Métrique', 'Valeur'])
                    # Forcer les types de colonnes pour compatibilité Arrow
                    if not metrics_df.empty:
                        metrics_df['Métrique'] = metrics_df['Métrique'].astype(str)
                        metrics_df['Valeur'] = metrics_df['Valeur'].astype(str)
                    st.dataframe(metrics_df)
                except Exception as e:
                    st.error(f"Erreur lors de la conversion des métriques simples en tableau : {e}")
                    st.write("Métriques simples (brutes) :")
                    st.json(metrics_for_table) # Fallback pour les métriques simples
        
            if remaining_complex_metrics:
                st.caption("Métriques complexes supplémentaires (non affichables dans le tableau simple) :")
                st.json(remaining_complex_metrics)
        
            # Si 'metrics' était vrai mais après filtrage tout est vide (très improbable)
            if not metrics_for_table and not remaining_complex_metrics:
                st.write("Aucune métrique brute à afficher en tableau (après filtrage).")
    else:
        st.warning("Aucune métrique de modèle récupérée.")

    st.header("Prédictions en Direct")
    with st.form("prediction_form"):
        st.subheader("Caractéristiques de l'accident:")
        input_features = {}
        cols_features = st.columns(3)
        for i, (feature_name, feature_info) in enumerate(FEATURE_DEFINITIONS.items()):
            with cols_features[i % 3]:
                # Fonction pour formater l'affichage des options dans le selectbox
                def format_options(option_value):
                    if "option_labels" in feature_info and option_value in feature_info["option_labels"]:
                        return feature_info["option_labels"][option_value]
                    return str(option_value) # Fallback si le libellé n'est pas défini

                input_features[feature_name] = st.selectbox(
                    label=feature_info["label"],
                    options=feature_info["options"],
                    index=feature_info["options"].index(feature_info["default"]) if feature_info["default"] in feature_info["options"] else 0,
                    format_func=format_options, # Utilisation de la fonction de formatage
                    help=feature_info.get("help", "")
                )
        submitted = st.form_submit_button("Prédire la Gravité")

    if submitted:
        prediction_result = predict_live(input_features)
        if prediction_result:
            st.subheader("Résultat de la Prédiction:")
            # L'API retourne maintenant une liste de prédictions sous la clé "predictions"
            pred_list = prediction_result.get("predictions") 

            if pred_list is not None and isinstance(pred_list, list) and len(pred_list) > 0:
                pred_val = pred_list[0] # Prendre la première prédiction de la liste
                try:
                    pred_int = int(pred_val) # Devrait être 0 ou 1
                    pred_label = MODEL_PREDICTION_LABEL_MAPPING.get(pred_int, "Prédiction Inconnue")
                    
                    # Adapter la couleur en fonction de la prédiction du modèle
                    # Par exemple: rouge pour 'Grave', vert pour 'Pas Grave'
                    color_map_model = {
                        0: "success", # Pas Grave
                        1: "error"    # Grave
                    }
                    getattr(st, color_map_model.get(pred_int, "info"))(f"Prédiction du modèle: **{pred_label}** (Code: {pred_int})")
                except ValueError: st.error(f"Prédiction reçue ({pred_val}) non valide.")
                st.caption("Réponse brute API:"); st.json(prediction_result)
            else:
                st.error("Clé 'predictions' non trouvée ou vide dans la réponse API.")
                st.caption("Réponse brute API:"); st.json(prediction_result)

    st.sidebar.header("À Propos")
    st.sidebar.info("Application Streamlit pour le projet MLOps Accidents 2023 de fin de formation DataScientest.")

else: # Message si non connecté
    st.info("Veuillez vous connecter pour accéder au contenu de l'application.")
