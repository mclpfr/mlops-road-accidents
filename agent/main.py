"""
FastAPI application for the MLOps agent "Gérard".
Provides a WebSocket interface and serves the HTML frontend.
"""
import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import uvicorn
from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import yaml

# Load configuration from YAML
with open('/app/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

server_config = config.get('agent_server', {})
agent_config = config.get('agent', {})

host = server_config.get('host', '0.0.0.0')
port = server_config.get('port', 8003)
web_concurrency = server_config.get('web_concurrency', 1)
external_url = agent_config.get('external_url', 'http://localhost:8003')

import docker
from llm_agent import load_api_key, answer_question
from docker_info_collector import collect_docker_info
from docker_info_collector import get_container_logs, _list_container_names

# Initialize FastAPI app
app = FastAPI(title="Gérard - Agent MLOps Autonome")

# Add CORS middleware to allow iframe embedding and WebSocket connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"]  # Expose all headers for WebSocket handshake
)

# Remove X-Frame-Options header to allow embedding in iframes
@app.middleware("http")
async def remove_frame_options(request, call_next):
    response = await call_next(request)
    # Some responses contain 'X-Frame-Options: DENY' by default, which blocks iframes
    if "x-frame-options" in response.headers:
        del response.headers["x-frame-options"]
    return response

# Create a directory for static files if it doesn't exist
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Health check endpoint
@app.get("/healthz")
async def health_check():
    """Health check endpoint for Docker and Kubernetes."""
    try:
        # Verify Docker daemon is accessible
        docker_client = docker.from_env()
        docker_client.ping()
        return {"status": "ok", "docker": "connected"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Docker daemon error: {str(e)}"}
        )

# Docker client
docker_client = docker.from_env()

def _list_container_names() -> list[str]:
    """Return the list of all container names (lower-case)."""
    try:
        return [c.name.lower() for c in docker_client.containers.list(all=True)]
    except Exception:
        return []

# Store active WebSocket connections
active_connections: List[WebSocket] = []

# Command registry
commands = {
    "logs": "Affiche les logs d'un conteneur (utiliser -f pour suivre en temps réel)",
    "stoplogs": "Arrête l'affichage en temps réel des logs",
    "restart": "Redémarre un conteneur",
    "status": "Affiche le statut d'un conteneur",
    "help": "Affiche l'aide des commandes disponibles",
}


class WebSocketManager:
    """Manages WebSocket connections and message broadcasting."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
    async def broadcast(self, message: str):
        """Send message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass
                
    async def send_to_client(self, websocket: WebSocket, message: str):
        """Send message to a specific client."""
        try:
            await websocket.send_text(message)
        except Exception:
            pass


# Initialize WebSocket manager
manager = WebSocketManager()

# Tasks streaming logs per (websocket, container_name)
log_stream_tasks: Dict[Tuple[WebSocket, str], asyncio.Task] = {}

async def stream_container_logs(container_name: str, websocket: WebSocket):
    """Stream Docker container logs to the websocket until cancelled."""
    try:
        # Find the container by exact or partial name
        containers = docker_client.containers.list(all=True)
        target = None
        for c in containers:
            if c.name.lower() == container_name.lower() or container_name.lower() in c.name.lower():
                target = c
                break
        if not target:
            return f"❌ Conteneur '{container_name}' non trouvé."
        
        # Stream logs continuously
        for line in target.logs(stream=True, follow=True, tail=0):
            await manager.send_to_client(websocket, line.decode(errors="ignore"))
    except asyncio.CancelledError:
        # Expected when the task is cancelled
        pass
    except Exception as e:
        await manager.send_to_client(websocket, f"❌ Erreur streaming logs: {str(e)}")


async def handle_command(command: str, websocket: WebSocket) -> Optional[str]:
    """Handle special commands prefixed with '!'."""
    if not command.startswith("!"):
        return None
        
    cmd_parts = command[1:].strip().split(maxsplit=1)
    cmd = cmd_parts[0].lower()
    arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

    # Fuzzy matching pour tolérer les fautes d'orthographe sur les commandes
    from difflib import get_close_matches
    ALL_COMMANDS = ["services", "logs", "stoplogs", "restart", "status", "help"]
    match = get_close_matches(cmd, ALL_COMMANDS, n=1, cutoff=0.7)
    if match:
        cmd = match[0]

    # Parse -f / --follow flags for log streaming
    def _parse_follow(arg_str: str):
        parts = arg_str.split()
        follow_flag = False
        for flag in ("-f", "--follow"):
            if flag in parts:
                follow_flag = True
                parts.remove(flag)
        return follow_flag, parts
    
    if cmd == "help":
        help_text = "📚 **Commandes disponibles:**\n\n"
        for cmd_name, desc in commands.items():
            help_text += f"- `!{cmd_name}` : {desc}\n"
        return help_text
        
    elif cmd == "logs":
        # Gestion du suivi temps réel avec -f / --follow
        follow, parts = _parse_follow(arg)
        if not parts:
            return "⚠️ Veuillez spécifier un nom de conteneur. Exemple: `!logs -f grafana`"

        container_name = parts[0]

        if follow:
            key = (websocket, container_name)
            # Stop any existing stream for this (ws, container)
            if key in log_stream_tasks:
                log_stream_tasks[key].cancel()
                del log_stream_tasks[key]
            task = asyncio.create_task(stream_container_logs(container_name, websocket))
            log_stream_tasks[key] = task
            return f"📡 Suivi en temps réel des logs pour '{container_name}' démarré. Tapez `!stoplogs {container_name}` pour arrêter."
        else:
            return f"🔍 Récupération des logs pour '{container_name}'..."
        
    elif cmd == "stoplogs":
        parts = arg.split()
        container_name = parts[0] if parts else None
        cancelled = False
        for key in list(log_stream_tasks.keys()):
            ws, cname = key
            if ws == websocket and (container_name is None or cname.lower() == container_name.lower()):
                log_stream_tasks[key].cancel()
                del log_stream_tasks[key]
                cancelled = True
        if cancelled:
            return "🛑 Flux de logs arrêté."
        else:
            return "⚠️ Aucun flux de logs en cours."

    elif cmd == "restart":
        if not arg:
            return "⚠️ Veuillez spécifier un nom de conteneur. Exemple: `!restart grafana`"
            
        return f"🔄 Tentative de redémarrage de '{arg}'..."
        
    elif cmd in ["services", "service", "servic", "servces", "serivce", "servies"]:
        import shlex, subprocess
        if not arg:
            return "⚠️ Veuillez spécifier au moins un groupe de services à démarrer. Exemple : `!services start-ui start-ml`."
        if arg.strip().lower() == "help":
            
            try:
                with open("/app/services.yml", "r") as f:
                    services_conf = yaml.safe_load(f)
                groups = services_conf.get("groups", {})
                commands = services_conf.get("commands", [])
                help_msg = "**Groupes de services disponibles :**\n"
                for g, desc in groups.items():
                    help_msg += f"- `{g}` : {desc}\n"
                help_msg += "\n**Commandes possibles :**\n"
                for c in commands:
                    help_msg += f"- `{c}`\n"
                help_msg += "\n**Exemples :**\n!services start-ui start-ml\n!services stop-ui\n!services restart-monitoring"
                return help_msg
            except Exception as e:
                return f"❌ Impossible de lire la configuration des services : {str(e)}"
        service_targets = shlex.split(arg)
        results = []
        for target in service_targets:
            try:
                result = subprocess.run(["make", target], capture_output=True, text=True, cwd="/home/ubuntu/mlops-road-accidents")
                if result.returncode == 0:
                    results.append(f"✅ `{target}` started successfully.")
                else:
                    results.append(f"❌ Error starting `{target}`:\n" + "```\n" + result.stderr.strip() + "\n```")
            except Exception as e:
                results.append(f"❌ Exception for `{target}`: {str(e)}")
        return "\n".join(results)

    elif cmd == "status":

        def humanize(bytes_val: int) -> str:
            return f"{bytes_val / (1024*1024):.0f}MiB" if bytes_val else "0MiB"

        if not arg:
            # Show status of all containers
            try:
                containers = docker_client.containers.list(all=True)
                headers = ["Conteneur", "État", "CPU %", "Mémoire", "Erreur"]
                md_lines = [
                    "| " + " | ".join(headers) + " |",
                    "|" + "|".join(["-"*len(h) for h in headers]) + "|",
                ]

                for c in containers:
                    status = c.status
                    cpu_pct, mem_combined = "-", "-"
                    if status == "running":
                        try:
                            stats = c.stats(stream=False)
                            cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - stats["precpu_stats"]["cpu_usage"]["total_usage"]
                            system_delta = stats["cpu_stats"].get("system_cpu_usage", 0) - stats["precpu_stats"].get("system_cpu_usage", 0)
                            if system_delta > 0:
                                cpu_pct_val = (cpu_delta / system_delta) * stats["cpu_stats"].get("online_cpus", 1) * 100.0
                                cpu_pct = f"{cpu_pct_val:.2f} %"
                            mem_used = stats["memory_stats"].get("usage", 0)
                            mem_limit = stats["memory_stats"].get("limit", 0)
                            mem_combined = f"{humanize(mem_used)} / {humanize(mem_limit) if mem_limit else 'Illimitée'}"
                        except Exception:
                            pass
                    
                    emoji = "🟢" if status == "running" else "🔴"
                    md_lines.append("| " + " | ".join([c.name, f"{emoji} {status.capitalize()}", cpu_pct, mem_combined, "Aucune"]) + " |")

                return "📊 **Statut des conteneurs :**\n\n" + "\n".join(md_lines) + "\n"
            except Exception as e:
                return f"❌ Erreur: {str(e)}"
            
        # Show status of specific container
        try:
            c = docker_client.containers.get(arg)
            status = c.status
            emoji = "🟢" if status == "running" else "🔴"
            stats_text = ""
            if status == "running":
                try:
                    stats = c.stats(stream=False)
                    cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - stats["precpu_stats"]["cpu_usage"]["total_usage"]
                    system_delta = stats["cpu_stats"].get("system_cpu_usage", 0) - stats["precpu_stats"].get("system_cpu_usage", 0)
                    if system_delta > 0:
                        cpu_pct = (cpu_delta / system_delta * stats["cpu_stats"].get("online_cpus", 1) * 100.0)
                        stats_text += f"\n- CPU: {cpu_pct:.2f}%"
                    mem_usage = stats["memory_stats"].get("usage", 0)
                    mem_limit = stats["memory_stats"].get("limit", 1)
                    stats_text += f"\n- Mémoire: {humanize(mem_usage)} / {humanize(mem_limit)}"
                except Exception:
                    stats_text = "\n- Statistiques non disponibles"
            return f"{emoji} **{c.name}**: {status}{stats_text}"
        except docker.errors.NotFound:
            return f"❌ Conteneur '{arg}' non trouvé."
        except Exception as e:
            return f"❌ Erreur: {str(e)}"

    # Command not recognized
    return None


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Interface HTML non trouvée")
    return FileResponse(str(html_path))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication with the agent."""
    await manager.connect(websocket)
    
    # Send welcome message
    await manager.send_to_client(
        websocket, 
        "👋 Bonjour, je suis Gérard, votre assistant MLOps. Comment puis-je vous aider aujourd'hui ?\n\n"
        "Vous pouvez me poser des questions sur l'état de la plateforme ou utiliser des commandes comme `!logs grafana`, `!status`, `!restart grafana`.\n\n"
        "Tapez `!help` pour voir toutes les commandes disponibles."
    )
    
    try:
        # Load API key once at connection time
        api_key = load_api_key()
        
        while True:
            message = await websocket.receive_text()
            
            # Détecter si c'est une simple salutation (mot unique ou début de message)
            salutations = ["bonjour", "salut", "hello", "coucou", "hey", "bonsoir"]
            message_clean = message.lower().strip().rstrip('!?.,;:')
            is_greeting = message_clean in salutations or any(message_clean.startswith(greet + " ") for greet in salutations)
            if is_greeting:
                await manager.send_to_client(websocket, "👋 Bonjour ! Comment puis-je vous aider ?")
                continue
            
            # Handle special commands
            response = await handle_command(message, websocket)
            if response:
                await manager.send_to_client(websocket, response)
                continue
                
            # Détecter les demandes d'affichage de logs en temps réel
            logs_patterns = [
                "affiche les logs en temps réel", "afficher les logs en temps réel",
                "montre les logs en temps réel", "montrer les logs en temps réel",
                "voir les logs en temps réel", "logs en temps réel",
                "affiche les logs", "afficher les logs", "montre les logs", "montrer les logs"
            ]
            
            is_logs_request = any(pattern in message.lower() for pattern in logs_patterns)
            if is_logs_request:
                # Récupérer la liste de tous les conteneurs disponibles
                try:
                    all_containers = docker_client.containers.list(all=True)
                    available_containers = [c.name for c in all_containers]
                    
                    # Extraire le nom du conteneur de la demande
                    container_name = None
                    
                    # Liste des mots-clés à rechercher dans la demande
                    container_keywords = [
                        "grafana", "postgres", "mlflow", "streamlit", "api", 
                        "prometheus", "loki", "alertmanager", "cadvisor", "promtail",
                        "evidently", "auth", "predict", "agent", "extract", "prepare", 
                        "import", "train", "synthet"
                    ]
                    
                    # Chercher un mot-clé dans le message
                    for keyword in container_keywords:
                        if keyword in message.lower():
                            # Trouver le conteneur correspondant
                            matching_containers = [c for c in available_containers if keyword in c.lower()]
                            if matching_containers:
                                container_name = matching_containers[0]
                                break
                    
                    # Si aucun conteneur spécifique n'est mentionné, demander à l'utilisateur
                    if not container_name:
                        container_list = "\n".join([f"- {c}" for c in available_containers])
                        await manager.send_to_client(websocket, f"⚠️ Veuillez spécifier un conteneur. Voici les conteneurs disponibles:\n{container_list}\n\nPour voir les logs, tapez par exemple: !logs -f nom_du_conteneur")
                        continue
                except Exception as e:
                    await manager.send_to_client(websocket, f"⚠️ Erreur lors de la récupération des conteneurs: {str(e)}")
                    continue
                
                # Vérifier que container_name est défini avant de continuer
                if container_name:
                    # Simuler la commande !logs -f container_name
                    await manager.send_to_client(websocket, f"📡 Je vais afficher les logs de {container_name} en temps réel...")
                    key = (websocket, container_name)
                    # Stop any existing stream for this (ws, container)
                    if key in log_stream_tasks:
                        log_stream_tasks[key].cancel()
                        del log_stream_tasks[key]
                    task = asyncio.create_task(stream_container_logs(container_name, websocket))
                    log_stream_tasks[key] = task
                    await manager.send_to_client(websocket, f"📡 Suivi en temps réel des logs pour '{container_name}' démarré. Tapez `!stoplogs {container_name}` pour arrêter.")
                else:
                    await manager.send_to_client(websocket, "⚠️ Aucun conteneur spécifié ou reconnu dans votre demande. Veuillez réessayer en mentionnant le nom du conteneur.")
                continue


            # Natural Language Processing for status requests
            message_lower = message.lower().strip()
            status_keywords = ["status", "état", "etat", "santé", "sante", "health", "plateforme", "lieux", "bilan", "aperçu"]

            # Check for general status request
            is_general_status_request = any(kw in message_lower for kw in status_keywords)
            all_container_names = _list_container_names()
            is_specific_container_mentioned = None
            words = [w for w in message_lower.replace("'", " ").split() if len(w) > 2]
            for cn in all_container_names:
                if cn in message_lower:
                    is_specific_container_mentioned = cn
                    break
                for w in words:
                    if w in cn:
                        is_specific_container_mentioned = cn
                        break
                if is_specific_container_mentioned:
                    break

            if is_general_status_request and is_specific_container_mentioned is None:
                response = await handle_command("!status", websocket)
                if response:
                    await manager.send_to_client(websocket, response)
                continue
            elif is_general_status_request and is_specific_container_mentioned:
                response = await handle_command(f"!status {is_specific_container_mentioned}", websocket)
                if response:
                    await manager.send_to_client(websocket, response)
                continue
                
            # Gérer les commandes de conteneur (démarrer, arrêter, redémarrer) par langage naturel
            # Détecte l'action (start / stop / restart)
            start_kw = ["démarre", "demarre", "start", "lance", "lancer"]
            stop_kw = ["arrête", "arrete", "stop", "stoppe", "stopper"]
            restart_kw = ["redémarre", "redemarre", "restart", "relance", "relancer"]

            if any(k in message_lower for k in start_kw + stop_kw + restart_kw):
                if any(k in message_lower for k in stop_kw):
                    action = "stop"
                    action_verb_present = "arrêté"
                elif any(k in message_lower for k in restart_kw):
                    action = "restart"
                    action_verb_present = "redémarré"
                else:
                    action = "start"
                    action_verb_present = "démarré"
                
                target_container = None
                try:
                    all_containers = [c.name for c in docker_client.containers.list(all=True)]
                    
                    # Recherche du nom de conteneur dans le message
                    for c_name in all_containers:
                        if c_name.lower() in message_lower:
                            target_container = c_name
                            break
                    
                    # Si aucun conteneur spécifique n'est trouvé, chercher des mots clés
                    if not target_container:
                        mots_message = message_lower.split()
                        for mot in mots_message:
                            if mot not in start_kw + stop_kw + restart_kw and len(mot) > 3:
                                search_key = mot
                                for c_name in all_containers:
                                    if search_key in c_name.lower():
                                        target_container = c_name
                                        break
                                if target_container:
                                    break
                    
                    if target_container:
                        await manager.send_to_client(websocket, f"⚙️ Tentative de {action} du conteneur {target_container}...")
                        container = docker_client.containers.get(target_container)
                        
                        try:
                            if action == "stop":
                                container.stop()
                                await manager.send_to_client(websocket, f"✅ Le conteneur {target_container} a été {action_verb_present} avec succès.")
                            elif action == "restart":
                                container.restart()
                                await manager.send_to_client(websocket, f"✅ Le conteneur {target_container} a été {action_verb_present} avec succès.")
                            else:  # start
                                container.start()
                                await manager.send_to_client(websocket, f"✅ Le conteneur {target_container} a été {action_verb_present} avec succès.")
                            continue
                        except Exception as e:
                            await manager.send_to_client(websocket, f"❌ Erreur lors de l'action {action} sur le conteneur {target_container} : {str(e)}")
                            continue
                    else:
                        await manager.send_to_client(websocket, f"❓ Je n'ai pas pu identifier quel conteneur vous souhaitez {action_verb_present}. Veuillez préciser le nom du conteneur.")
                        continue
                except Exception as e:
                    await manager.send_to_client(websocket, f"❌ Erreur lors de la recherche du conteneur : {str(e)}")
                    continue

            # Si aucune commande spéciale ou mot-clé n'est détecté, envoyer au LLM
            await manager.send_to_client(websocket, "🧠 Je traite votre demande...")
            docker_info = collect_docker_info()  # Collect fresh data
            answer = answer_question(message, docker_info, api_key)
            await manager.send_to_client(websocket, answer)

            # Gérer les commandes de conteneur (démarrer, arrêter, redémarrer) par langage naturel
            message_lower = message.lower().strip()
            # Détecte l'action (start / stop / restart)
            start_kw = ["démarre", "demarre", "start"]
            stop_kw = ["arrête", "arrete", "stop"]
            restart_kw = ["redémarre", "redemarre", "restart"]

            if any(k in message_lower for k in start_kw + stop_kw + restart_kw):
                if any(k in message_lower for k in stop_kw):
                    action = "stop"
                    action_verb_present = "arrêté"
                elif any(k in message_lower for k in restart_kw):
                    action = "restart"
                    action_verb_present = "redémarré"
                else:
                    action = "start"
                    action_verb_present = "démarré"
                
                target_container = None
                try:
                    all_containers = [c.name for c in docker_client.containers.list(all=True)]
                    mots_message = message_lower.split()
                    for mot in mots_message:
                        if mot not in start_kw + stop_kw + restart_kw:
                            search_key = mot
                            break
                    else:
                        search_key = ""
                    for c_name in all_containers:
                        if search_key and search_key in c_name.lower():
                            target_container = c_name
                            break
                    if target_container:
                        await manager.send_to_client(websocket, f"⚙️ Tentative de démarrage du conteneur {target_container}...")
                        container = docker_client.containers.get(target_container)
                        if action == "stop":
                            container.stop()
                        elif action == "restart":
                            container.restart()
                        else:
                            import subprocess
                            result = subprocess.run(["docker", "start", target_container], capture_output=True, text=True)
                            if result.returncode == 0:
                                await manager.send_to_client(websocket, f"✅ Le conteneur {target_container} a été {action_verb_present} avec succès.")
                            else:
                                await manager.send_to_client(websocket, f"❌ Erreur lors du démarrage du conteneur {target_container} : {result.stderr.strip()}")
                        continue # On ne passe pas la main au LLM
                    else:
                        # Aucun conteneur trouvé, proposer la création
                        await manager.send_to_client(websocket, '''❌ Aucun conteneur Loki existant n'a été trouvé.

Vous pouvez en créer un avec la commande suivante :

```
docker run -d --name loki grafana/loki:2.9.0
```
''')
                        continue # On ne passe pas la main au LLM

                except docker.errors.NotFound:
                    if target_container:
                        await manager.send_to_client(websocket, f"⚠️ Conteneur '{target_container}' non trouvé.")
                    # Si aucun conteneur trouvé, on laisse le LLM répondre
                    pass
                except Exception as e:
                    await manager.send_to_client(websocket, f"❌ Erreur lors de l'opération sur le conteneur : {str(e)}")
                    continue

            # Afficher un message d'attente uniquement pour les requêtes techniques
            message_lower = message.lower().strip()
            question_starters = ["qui", "que", "quoi", "quand", "où", "comment", "pourquoi", "quel", "est-ce que"]
            is_a_question = message_lower.endswith('?') or any(message_lower.startswith(starter) for starter in question_starters)

            await manager.send_to_client(websocket, "🧠 Je réfléchis à votre question..." if is_a_question else "🧠 Je traite votre demande...")
            
            # Pour les salutations, utiliser un contexte vide
            # Pour les questions techniques, collecter les informations Docker
            if is_greeting:
                context = ""
            else:
                # Gather context for LLM
                docker_info = collect_docker_info()
                context = "\n\n".join([f"### {k} ###\n{v}" for k, v in docker_info.items()])
                # Hard limit ~15000 chars to stay within Together token limit
                context = context[:15000]
            
            try:
                # Query LLM with user's question
                answer = answer_question(message, context, api_key)
                
                # Add a bit of personality to the response, but only for technical questions (not for greetings)
                if not is_greeting:
                    personality_prefixes = [
                        "D'après mon analyse, ",
                        "Voici ce que j'observe : ",
                        "Après vérification, ",
                        "Je viens de regarder et ",
                        "Si je comprends bien la situation, "
                    ]
                    import random
                    if not answer.startswith(("Je ", "J'", "Voici", "D'après", "Après", "Selon", "Bonjour", "Salut", "Hello")):
                        answer = f"{random.choice(personality_prefixes)}{answer}"
                # Pour les salutations, on laisse la réponse telle quelle
                
                await manager.send_to_client(websocket, answer)
            except Exception as e:
                await manager.send_to_client(websocket, f"❌ Désolé, j'ai rencontré une erreur: {str(e)}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        # Annuler les flux de logs associés à cette connexion
        for key in list(log_stream_tasks):
            if key[0] == websocket:
                log_stream_tasks[key].cancel()
                del log_stream_tasks[key]
    except Exception as e:
        print(f"Error in websocket connection: {str(e)}")
        manager.disconnect(websocket)
        for key in list(log_stream_tasks):
            if key[0] == websocket:
                log_stream_tasks[key].cancel()
                del log_stream_tasks[key]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
