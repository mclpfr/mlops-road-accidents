import os
import json
import glob
import yaml
import requests
from docker_info_collector import collect_docker_info
from pathlib import Path
from typing import List

TOGETHER_ENDPOINT = os.getenv("TOGETHER_ENDPOINT", "https://api.together.xyz/v1/chat/completions")
MODEL_NAME = os.getenv("TOGETHER_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
LOG_DIR = os.getenv("LOG_DIR", "./agent/logs")

SYSTEM_PROMPT = """
Tu es un agent IA expert en MLOps et Docker. On te fournit les métriques et informations système d'une plateforme basée sur Docker Compose.

Ta mission est d’analyser ces données et de répondre à des questions posées en langage naturel par un humain. Tu peux répondre sur :
- consommation CPU/mémoire
- erreurs critiques
- conteneurs arrêtés ou en erreur
- IO réseau ou disque
- stabilité de la plateforme
- logs des conteneurs

Tu dois :
- Être concis, clair et utile
- Demander plus de contexte si nécessaire
"""

def load_api_key() -> str:
    """Return the Together.ai API key from *config.yaml* (key: `together_api_key`) or env var as fallback."""

    config_path = Path("config.yaml")
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if isinstance(cfg, dict):
                    api_key = cfg.get("together_api_key")
                    if api_key:
                        return api_key
        except Exception as exc:
            print(f"Could not parse config.yaml: {exc}")

    # Fallback to environment variable if not present in config file
    api_key = os.getenv("TOGETHER_API_KEY")
    if api_key:
        return api_key

    raise RuntimeError("Clé Together.ai non trouvée (config.yaml → together_api_key ou variable d'environnement TOGETHER_API_KEY)")


def tail_file(path: Path, max_lines: int = 200) -> List[str]:
    """Return the last *max_lines* lines from *path*."""
    try:
        with path.open("r", errors="ignore") as f:
            return f.readlines()[-max_lines:]
    except Exception:
        return []


def gather_context() -> str:
    """Collect diagnostics from Docker and format as a single text block for the LLM."""
    docker_info = collect_docker_info()
    context = "\n\n".join([f"### {k} ###\n{v}" for k, v in docker_info.items()])
    # Hard limit ~15000 chars to stay within Together token limit
    return context[:15000]


def query_llm(question: str, context: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Contexte:\n{context}\n\nQuestion:\n{question}",
        },
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 512,
        "top_p": 0.9,
    }

    response = requests.post(TOGETHER_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Together.ai API error {response.status_code}: {response.text}")

    data = response.json()
    # The schema is {"choices": [{"message": {"role": "assistant", "content": "..."}}]}
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected API response format: {data}")


def main():
    api_key = load_api_key()
    print("\n🔧 Agent de diagnostic MLOps démarré. Posez votre question (Ctrl+C pour quitter).\n")

    while True:
        try:
            question = input("❓ Votre question : ")
        except KeyboardInterrupt:
            print("\n👋 Arrêt de l'agent. À bientôt !")
            break
        except EOFError:
            # Stdin fermé (mode détaché). On attend avant de réessayer pour éviter l'arrêt du conteneur.
            import time
            print("STDIN fermé – attente...")
            time.sleep(60)
            continue
        if not question.strip():
            continue

        context = gather_context()
        try:
            answer = query_llm(question, context, api_key)
            print(f"\n💡 Réponse de l'agent :\n{answer}\n")
        except Exception as exc:
            print(f"⚠️  Erreur lors de l'appel au modèle : {exc}\n")


if __name__ == "__main__":
    main()
