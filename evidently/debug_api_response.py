import requests

# URL de l'API Evidently
url = "http://localhost:8001/drift_score"

# Faire une requête pour obtenir le score de dérive
response = requests.get(url)

# Afficher les informations de débogage
print("=== Debug API Response ===")
print(f"Status Code: {response.status_code}")
print(f"Headers: {response.headers}")
print("\nRaw Response Content:")
print(response.text)

# Essayer d'extraire le score
try:
    lines = response.text.split('\n')
    print("\nResponse Lines:", lines)
    if len(lines) >= 3:
        print(f"\nScore de dérive: {lines[2]}")
    else:
        print("Réponse inattendue, nombre de lignes insuffisant")
except Exception as e:
    print(f"\nErreur lors de l'analyse de la réponse: {e}")
