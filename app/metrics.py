# app/metrics.py - À ajouter à ton projet
import time
import requests
import streamlit as st
from datetime import datetime
from typing import Dict, Any
import threading
import queue
from functools import wraps

class StreamlitMetricsCollector:
    """Collecteur de métriques pour Streamlit avec envoi vers Prometheus Pushgateway"""
    
    def __init__(self):
        # Configuration Prometheus Pushgateway via Grafana Cloud
        self.pushgateway_url = st.secrets.get("GRAFANA_PROMETHEUS_URL", "").rstrip('/') + "/api/v1/write"
        self.username = st.secrets.get("GRAFANA_CLOUD_USER", "")
        self.api_key = st.secrets.get("GRAFANA_CLOUD_API_KEY", "")
        self.job_name = "streamlit_road_accidents"
        
        # Métriques en mémoire
        self.metrics = {
            'page_views': {},
            'predictions': {},
            'model_load_time': [],
            'prediction_time': [],
            'errors': {},
            'drift_score': 0.0
        }
        
        # Configuration de l'authentification
        self.auth = (self.username, self.api_key) if self.username and self.api_key else None
        self.headers = {
            'Content-Type': 'text/plain',
            'X-Prometheus-Remote-Write-Version': '0.1.0'
        }
        
        # Queue pour l'envoi asynchrone
        self.queue = queue.Queue()
        self.running = True
        
        # Thread d'envoi
        self.sender_thread = threading.Thread(target=self._metric_sender, daemon=True)
        self.sender_thread.start()
        
        # Enregistrement du démarrage
        self._push_metric("app_started_total", 1, metric_type="counter")
    
    def _push_metric(self, name, value, labels=None, metric_type="gauge"):
        """Envoie une métrique unique au Pushgateway"""
        if not self.pushgateway_url or not self.auth:
            return False
            
        try:
            # Formatage des labels
            if labels is None:
                labels = {}
            
            # Ajout des labels communs
            labels.update({
                "job": self.job_name,
                "app": "road_accidents",
                "environment": "production"
            })
            
            # Formatage de la métrique
            metric_data = self._format_metric(name, value, labels, metric_type)
            
            # Envoi de la métrique
            response = requests.post(
                self.pushgateway_url,
                data=metric_data,
                auth=self.auth,
                headers=self.headers,
                timeout=5
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Erreur lors de l'envoi de la métrique {name}: {str(e)}")
            return False
    
    def track_metric(self, name, value, labels=None):
        """
        Enregistre une métrique personnalisée
        
        Args:
            name: Nom de la métrique
            value: Valeur de la métrique
            labels: Dictionnaire de labels pour la métrique
        """
        if labels is None:
            labels = {}
            
        # Ajout des labels communs
        labels.update({
            "job": self.job_name,
            "app": "road_accidents",
            "environment": "production"
        })
        
        # Enregistrement dans la file d'attente
        self.queue.put(('gauge', name, value, labels))
    
    def track_page_view(self, page_name: str):
        """Enregistre une vue de page"""
        key = 'streamlit_page_views_total'
        labels = {"page": page_name}
        self.metrics['page_views'][key] = self.metrics['page_views'].get(key, 0) + 1
        self.queue.put(('counter', key, 1, labels))
    
    def track_prediction(self, model_version: str, prediction_result: str, duration: float):
        """Enregistre une prédiction"""
        # Compteur de prédictions
        key = 'streamlit_predictions_total'
        labels = {
            "model_version": model_version,
            "result": prediction_result
        }
        self.metrics['predictions'][key] = self.metrics['predictions'].get(key, 0) + 1
        self.queue.put(('counter', key, 1, labels))
        
        # Temps de prédiction
        if prediction_result == 'success':
            key = 'streamlit_prediction_duration_seconds'
            labels = {"model_version": model_version}
            self.metrics['prediction_time'].append(duration)
            self.queue.put(('histogram', key, [duration], labels))
    
    def track_model_load(self, duration: float, success: bool):
        """Enregistre le chargement du modèle"""
        status = "success" if success else "failed"
        key = f'streamlit_model_loads_total{{status="{status}"}}'
        self.queue.put(('counter', key, 1))
        
        if success:
            self.metrics['model_load_time'].append(duration)
            self.queue.put(('histogram', 'streamlit_model_load_duration_seconds', duration))
    
    def update_drift_score(self, score: float):
        """Met à jour le score de drift"""
        self.metrics['drift_score'] = score
        self.queue.put(('gauge', 'ml_data_drift_score', score))
    
    def track_error(self, error_type: str, message: str = ""):
        """Enregistre une erreur"""
        # Compteur d'erreurs par type
        key = 'streamlit_errors_total'
        labels = {"type": error_type}
        self.metrics['errors'][key] = self.metrics['errors'].get(key, 0) + 1
        self.queue.put(('counter', key, 1, labels))
        
        # On peut aussi logger le message d'erreur si nécessaire
        if message:
            error_key = 'streamlit_error_messages'
            error_labels = {
                "type": error_type,
                "message": message[:100]  # Tronquer le message si trop long
            }
            self.queue.put(('gauge', error_key, 1, error_labels))
    
    def _metric_sender(self):
        """Thread qui envoie les métriques au Pushgateway Prometheus"""
        batch = []
        last_send = time.time()
        
        while self.running:
            try:
                # Récupère une métrique avec un timeout
                try:
                    metric = self.queue.get(timeout=5)
                    batch.append(metric)
                except queue.Empty:
                    pass
                
                # Envoie le batch toutes les 30 secondes ou s'il dépasse 100 métriques
                if batch and (len(batch) >= 100 or time.time() - last_send >= 30):
                    self._send_metrics(batch)
                    batch = []
                    last_send = time.time()
                    
            except Exception as e:
                print(f"Erreur dans le thread d'envoi: {e}")
                time.sleep(5)
    
    def _send_metrics(self, metrics_batch):
        """Envoi des métriques par lots au format Prometheus"""
        if not self.pushgateway_url or not self.api_key:
            return
            
        try:
            # Formatage des métriques pour Prometheus
            metrics_data = []
            
            for metric_type, name, value, *extra in metrics_batch:
                labels = extra[0] if extra and isinstance(extra[0], dict) else {}
                
                if metric_type == 'counter':
                    metrics_data.append(self._format_metric(name, value, labels, "counter"))
                elif metric_type == 'gauge':
                    metrics_data.append(self._format_metric(name, value, labels, "gauge"))
                elif metric_type == 'histogram':
                    # Pour les histogrammes, on crée plusieurs métriques
                    if not isinstance(value, (list, tuple)):
                        value = [value]
                    
                    # Compteur
                    metrics_data.append(self._format_metric(
                        f"{name}_count", 
                        len(value), 
                        labels
                    ))
                    # Somme
                    metrics_data.append(self._format_metric(
                        f"{name}_sum", 
                        sum(value), 
                        labels
                    ))
                    # Quantiles (pour un histogramme simple)
                    if value:
                        metrics_data.append(self._format_metric(
                            f"{name}_bucket", 
                            len([x for x in value if x <= 0.5]), 
                            {**labels, "le": "0.5"}
                        ))
                        metrics_data.append(self._format_metric(
                            f"{name}_bucket", 
                            len([x for x in value if x <= 0.9]), 
                            {**labels, "le": "0.9"}
                        ))
                        metrics_data.append(self._format_metric(
                            f"{name}_bucket", 
                            len([x for x in value if x <= 0.99]), 
                            {**labels, "le": "0.99"}
                        ))
                        metrics_data.append(self._format_metric(
                            f"{name}_bucket", 
                            len(value), 
                            {**labels, "le": "+Inf"}
                        ))
            
            # Envoi des métriques
            if metrics_data:
                response = requests.post(
                    self.pushgateway_url,
                    data=''.join(metrics_data),
                    auth=(self.username, self.api_key),
                    headers={'Content-Type': 'text/plain'},
                    timeout=10
                )
                response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            print(f"Erreur réseau lors de l'envoi des métriques: {str(e)}")
        except Exception as e:
            print(f"Erreur lors de l'envoi des métriques: {str(e)}")
    
    def _format_metric(self, name, value, labels, metric_type):
        """Formatage d'une métrique Prometheus"""
        metric = f"{name} {value}"
        
        if labels:
            metric += "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
        
        metric += f" {int(time.time())}\n"
        
        return metric

# Instance globale
metrics_collector = StreamlitMetricsCollector()

# Décorateurs utiles
def track_execution_time(metric_name: str):
    """Décorateur pour mesurer le temps d'exécution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.queue.put(('histogram', f'{metric_name}_duration_seconds', duration))
                return result
            except Exception as e:
                metrics_collector.track_error(metric_name, str(e))
                raise
        return wrapper
    return decorator

# Modifications à apporter dans main.py
def integrate_metrics_in_main():
    """
    Ajoute ces modifications dans ton main.py
    """
    
    # 1. Import au début
    from metrics import metrics_collector, track_execution_time
    
    # 2. Modifier load_local_model pour tracker le chargement
    @st.cache_resource(ttl=3600)
    @track_execution_time('model_load')
    def load_local_model():
        """Télécharge et met en cache le meilleur modèle enregistré dans MLflow."""
        start_time = time.time()
        try:
            if _find_best_model is None:
                st.error("Impossible d'importer la fonction find_best_model pour charger le modèle.")
                metrics_collector.track_model_load(0, False)
                return None
            
            model, _ = _find_best_model()
            duration = time.time() - start_time
            metrics_collector.track_model_load(duration, True)
            return model
        except Exception as e:
            duration = time.time() - start_time
            metrics_collector.track_model_load(duration, False)
            metrics_collector.track_error('model_load', str(e))
            st.error(f"Erreur lors du chargement du modèle : {e}")
            return None
    
    # 3. Dans la fonction main(), tracker les pages vues
    def main(accidents_count):
        # ... code existant ...
        
        # Tracker la page vue
        metrics_collector.track_page_view(page)
        
        # ... reste du code ...
    
    # 4. Dans show_interactive_demo(), tracker les prédictions
    def track_prediction_in_demo():
        if submitted:
            start_time = time.time()
            
            # ... code de mapping existant ...
            
            try:
                # Prédiction
                pred = int(model.predict(X)[0])
                confidence = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    confidence = proba[pred]
                
                # Tracker la prédiction
                duration = time.time() - start_time
                prediction_result = "grave" if pred == 0 else "pas_grave"
                model_version = best_version.version if best_version else "unknown"
                
                metrics_collector.track_prediction(model_version, prediction_result, duration)
                
                # ... affichage du résultat ...
                
            except Exception as e:
                metrics_collector.track_error('prediction', str(e))
                st.error(f"Erreur lors de la prédiction: {e}")
    
    # 5. Ajouter un tracker de drift périodique
    def track_drift_score():
        if 'last_drift_update' not in st.session_state:
            st.session_state.last_drift_update = time.time()
            st.session_state.current_drift = 0.15
        
        # Mise à jour toutes les 5 minutes
        if time.time() - st.session_state.last_drift_update > 300:
            # Simuler un drift (remplacer par ton calcul réel)
            import random
            drift = st.session_state.current_drift + random.uniform(-0.02, 0.03)
            drift = max(0, min(1, drift))
            
            metrics_collector.update_drift_score(drift)
            st.session_state.current_drift = drift
            st.session_state.last_drift_update = time.time()
    
    # 6. Appeler track_drift_score dans le main
    if __name__ == "__main__":
        # ... code existant ...
        
        # Tracker le drift
        track_drift_score()
        
        # ... reste du code ...
