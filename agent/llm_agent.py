"""
LLM Agent module for Gérard.
Encapsulates all interactions with the Together.ai API.
"""
import os
import json
import yaml
import requests
from pathlib import Path
from typing import Dict, Optional, List, Any

# Configuration
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions")
MODEL_NAME = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")

# System prompt for the agent
SYSTEM_PROMPT = """
Tu es Gérard, un agent IA spécialisé EXCLUSIVEMENT en MLOps et Docker. Tu surveilles une plateforme basée sur Docker Compose.

RESTRICTIONS STRICTES :
- Tu ne dois JAMAIS discuter de sujets qui ne sont pas liés à MLOps, Docker, conteneurs, infrastructure IT, ou gestion de plateforme
- Tu ne dois JAMAIS fournir d'informations sur des sujets comme la cuisine, le sport, les voyages, la médecine, ou tout autre sujet non lié à ta spécialité
- Si on te pose une question hors sujet, réponds poliment que tu es un agent MLOps spécialisé et que tu ne peux répondre qu'à des questions liées à ta spécialité

IMPORTANT :
- Tu es autorisé à afficher des tableaux texte (ASCII) pour présenter des informations système, des statuts, des consommations de ressources, etc.
- N'indique jamais que tu ne peux pas afficher de tableau si tu es capable d'en générer un en texte.
- Quand tu présentes des données tabulaires (statuts, ressources, etc.), utilise TOUJOURS le format Markdown de tableau (avec | et ---). N'utilise jamais de listes à puces ou de blocs code pour cela.
- Si on te demande l'état de la plateforme ou un statut général, ne génère PAS de tableau toi-même. Réponds simplement que tu utilises la commande interne pour afficher le statut.

Ta mission principale est d'aider l'utilisateur avec la surveillance et la gestion de sa plateforme MLOps. Tu dois :

1. Être capable de faire du "small talk" UNIQUEMENT dans le contexte professionnel MLOps :
   - Répondre de manière appropriée aux salutations ("bonjour", "salut", etc.) sans fournir immédiatement un diagnostic technique
   - Engager une conversation amicale mais toujours professionnelle et centrée sur MLOps
   - Comprendre quand l'utilisateur veut simplement discuter vs quand il demande une analyse technique

2. Analyser les données système UNIQUEMENT quand on te le demande explicitement ou quand c'est pertinent dans le contexte de la conversation :
   - Consommation CPU/mémoire des conteneurs
   - Erreurs critiques dans les logs
   - Conteneurs arrêtés ou en erreur
   - Stabilité de la plateforme
   - Logs des conteneurs

3. Adopter un ton professionnel mais amical :
   - Être concis mais informatif
   - Parler comme un collègue DevOps (sans exagérer)
   - Utiliser uniquement les données fournies dans le contexte pour répondre aux questions techniques
   - Demander plus de contexte si nécessaire pour répondre précisément

Exemples de réponses appropriées :
- À "Bonjour" : "Bonjour ! Comment puis-je vous aider avec votre plateforme MLOps aujourd'hui ?"
- À "Comment ça va ?" : "Très bien, merci ! Prêt à vous assister avec votre infrastructure Docker. Que puis-je faire pour vous ?"
- À "Quel est l'état des conteneurs ?" : "Je vois que tous les conteneurs sont en cours d'exécution. Grafana consomme un peu plus de CPU que d'habitude. Souhaitez-vous des détails spécifiques ?"
- À "Donne-moi une recette de cuisine" : "Désolé, je suis un agent spécialisé en MLOps et Docker. Je ne peux pas vous fournir de recettes de cuisine. En revanche, je peux vous aider avec la gestion de vos conteneurs Docker ou d'autres questions liées à votre infrastructure."

Important : 
- N'analyse pas automatiquement l'état du système sans qu'on te le demande explicitement
- Ne réponds JAMAIS à des questions hors sujet, même si elles semblent simples ou inoffensives
"""


def load_api_key() -> str:
    """Return the LLM API key.

    Recherche dans l'ordre :
    1. Variable d'environnement ``LLM_API_KEY`` (recommandée pour l'exécution en conteneur).
    2. Fichiers ``config.yaml`` courants (``/app/config.yaml``, ``/config.yaml``, racine du repo).

    Le fichier YAML attendu doit contenir :

    ```yaml
    agent:
      llm_api_key: "sk-..."
    ```
    """

    # 1. Environment variable
    env_key = os.getenv("LLM_API_KEY")
    if env_key:
        return env_key.strip()

    # 2. Possible config locations
    possible_paths = [
        Path("/app/config.yaml"),
        Path("/config.yaml"),
        (Path(__file__).parent.parent / "config.yaml"),  # racine du repo montée en volume
    ]

    for cfg_path in possible_paths:
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Détection simple d'un fichier Ansible Vault chiffré
            if content.lstrip().startswith("$ANSIBLE_VAULT"):
                raise ValueError(
                    f"Le fichier {cfg_path} est chiffré (Ansible Vault). Définissez la clé via la variable d'environnement LLM_API_KEY ou fournissez un YAML en clair."
                )
            config = yaml.safe_load(content) or {}
            if isinstance(config, dict):
                api_key = (
                    config.get("agent", {}).get("llm_api_key")
                    or config.get("llm_api_key")
                )
                if api_key:
                    print(f"Clé API chargée depuis {cfg_path}.")
                    return str(api_key).strip()
        except yaml.YAMLError as e:
            # Essayer le chemin suivant si le YAML est invalide
            print(f"⚠️ YAML invalide dans {cfg_path}: {e}. On continue la recherche…")
            continue
        except Exception as e:
            # Journaliser puis continuer
            print(f"⚠️ Impossible de lire {cfg_path}: {e}. On continue la recherche…")
            continue

    # Si aucune clé trouvée
    raise ValueError(
        "Clé API non trouvée. Définissez LLM_API_KEY ou fournissez un fichier config.yaml contenant agent.llm_api_key."
    )


def answer_question(question: str, context: str, api_key: Optional[str] = None) -> str:
    """
    Query the LLM with a question and context.
    
    Args:
        question: The user's question
        context: System context (Docker stats, logs, etc.)
        api_key: LLM API key (will be loaded if None)
        
    Returns:
        The LLM's response
    """
    if api_key is None:
        api_key = load_api_key()
    
    # Mode démo - retourner des réponses prédéfinies sans appeler l'API
    if api_key == "demo":
        import random
        
        # Extraire des informations du contexte pour des réponses plus pertinentes
        containers_running = "running" in context.lower()
        has_errors = "error" in context.lower() or "erreur" in context.lower()
        high_cpu = "cpu: 8" in context.lower() or "cpu: 9" in context.lower()
        
        # Réponses génériques basées sur le contexte et la question
        if "grafana" in question.lower() or "dashboard" in question.lower():
            return "Grafana semble fonctionner normalement. Les dashboards sont accessibles et la consommation de ressources est stable."
        elif "postgres" in question.lower() or "base de données" in question.lower() or "database" in question.lower():
            return "La base de données PostgreSQL est active et répond aux requêtes. Je ne détecte pas de problèmes de performance."
        elif "mémoire" in question.lower() or "memory" in question.lower():
            if high_cpu:
                return "Je détecte une consommation élevée de mémoire sur certains conteneurs. Particulièrement le service de prédiction qui utilise plus de 70% de sa mémoire allouée."
            else:
                return "La consommation de mémoire est normale sur tous les conteneurs. Aucun problème détecté."
        elif "cpu" in question.lower():
            if high_cpu:
                return "La charge CPU est élevée sur le conteneur de prédiction. Il utilise actuellement plus de 80% de ses ressources allouées."
            else:
                return "La charge CPU est normale sur tous les conteneurs. Tout fonctionne efficacement."
        elif "problème" in question.lower() or "erreur" in question.lower() or "error" in question.lower():
            if has_errors:
                return "J'ai détecté quelques erreurs dans les logs du service de prédiction. Il semble y avoir des problèmes de connexion intermittents avec la base de données."
            else:
                return "Je ne détecte pas d'erreurs critiques dans les logs des conteneurs. Tout semble fonctionner normalement."
        elif "status" in question.lower() or "état" in question.lower() or "santé" in question.lower():
            if containers_running:
                return "Tous les conteneurs sont en cours d'exécution. La plateforme MLOps est stable et opérationnelle."
            else:
                return "Certains conteneurs ne sont pas en cours d'exécution. Vous devriez vérifier l'état des services critiques."
        else:
            # Réponses génériques si aucune correspondance spécifique
            generic_responses = [
                "D'après mon analyse des conteneurs Docker, tout semble fonctionner normalement. Les services sont stables et répondent correctement.",
                "Je ne détecte pas de problèmes particuliers dans l'infrastructure. Les métriques sont dans les normes attendues.",
                "L'infrastructure MLOps fonctionne correctement. Les services de monitoring et de prédiction sont opérationnels.",
                "Les conteneurs Docker sont en bonne santé. Je ne vois pas d'anomalies dans les logs ou les métriques de performance.",
                "La plateforme est stable. Les services communiquent correctement entre eux et les ressources système sont bien réparties."
            ]
            return random.choice(generic_responses)
    
    # Mode API réelle - appel à OpenRouter
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # En-têtes spécifiques à OpenRouter
        "HTTP-Referer": "https://mlops-road-accidents.com",
        "X-Title": "Gérard MLOps Agent"
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Contexte:\n{context}\n\nQuestion:\n{question}\n\nConsigne: Réponds de façon concise (max 10 lignes) et synthétique en français.",
        },
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.3,  # Lower temperature for more consistent responses
        "max_tokens": 256,
        "top_p": 0.9,
        "stream": False,
        "response_format": {"type": "text"}
    }

    try:
        response = requests.post(LLM_ENDPOINT, headers=headers, json=payload, timeout=60)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"Erreur de communication avec l'API OpenRouter: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"Format de réponse inattendu de l'API: {str(e)}"
    except Exception as e:
        return f"Erreur inattendue: {str(e)}"


# For direct testing
if __name__ == "__main__":
    from docker_info_collector import collect_docker_info
    
    api_key = load_api_key()
    context = "\n\n".join([f"### {k} ###\n{v}" for k, v in collect_docker_info().items()])
    
    while True:
        question = input("Question: ")
        if not question:
            break
            
        answer = answer_question(question, context, api_key)
        print(f"\nRéponse: {answer}\n")
