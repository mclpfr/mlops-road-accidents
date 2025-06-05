import requests
import numpy as np
import pandas as pd
import time

def test_drift_variation():
    # URL de l'API Evidently
    url = "http://localhost:8001/drift_score"
    
    print("=== Test de variation du score de dérive ===\n")
    
    # Niveaux de bruit à tester
    noise_levels = [0.1, 0.3, 0.5, 0.7, 1.0]
    
    # Faire une première requête pour le score initial
    try:
        response = requests.get(url)
        initial_score = float(response.text.split('\n')[-1].split()[-1])
        print(f"Score de dérive initial: {initial_score:.2f}\n")
    except Exception as e:
        print(f"Erreur lors de la récupération du score initial: {e}")
        return
    
    # Tester différents niveaux de bruit
    for noise in noise_levels:
        try:
            # Faire une requête avec un paramètre pour simuler différents niveaux de dérive
            params = {'noise': noise}
            response = requests.get(url, params=params)
            
            # Extraire le score de la réponse
            score = float(response.text.split('\n')[-1].split()[-1])
            print(f"Niveau de bruit: {noise:.1f} - Score de dérive: {score:.2f}")
            
            # Attendre un court instant entre les requêtes
            time.sleep(1)
            
        except Exception as e:
            print(f"Erreur avec le niveau de bruit {noise}: {e}")
    
    print("\n=== Test terminé ===")

if __name__ == "__main__":
    test_drift_variation()
