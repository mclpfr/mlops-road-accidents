"""
FastAPI application for the MLOps agent "Gérard".
Provides a WebSocket interface and serves the HTML frontend.
"""
import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

import docker
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from llm_agent import load_api_key, answer_question
from docker_info_collector import collect_docker_info
from docker_info_collector import get_container_logs, _list_container_names

# Initialize FastAPI app
app = FastAPI(title="Gérard - Agent MLOps Autonome")

# Add CORS middleware to allow iframe embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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

# Docker client
docker_client = docker.from_env()

# Store active WebSocket connections
active_connections: List[WebSocket] = []

# Command registry
commands = {
    "logs": "Affiche les logs d'un conteneur",
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


async def handle_command(command: str, websocket: WebSocket) -> bool:
    """Handle special commands prefixed with '!'."""
    if not command.startswith("!"):
        return False
        
    cmd_parts = command[1:].strip().split(maxsplit=1)
    cmd = cmd_parts[0].lower()
    arg = cmd_parts[1] if len(cmd_parts) > 1 else ""
    
    if cmd == "help":
        help_text = "📚 **Commandes disponibles:**\n\n"
        for cmd_name, desc in commands.items():
            help_text += f"- `!{cmd_name}` : {desc}\n"
        await manager.send_to_client(websocket, help_text)
        return True
        
    elif cmd == "logs":
        if not arg:
            await manager.send_to_client(websocket, "⚠️ Veuillez spécifier un nom de conteneur. Exemple: `!logs grafana`")
            return True
            
        await manager.send_to_client(websocket, f"🔍 Récupération des logs pour '{arg}'...")
        logs = get_container_logs(arg, tail=10)
        formatted_logs = f"📜 **Logs pour {arg}:**\n```\n{logs}\n```"
        await manager.send_to_client(websocket, formatted_logs)
        return True
        
    elif cmd == "restart":
        if not arg:
            await manager.send_to_client(websocket, "⚠️ Veuillez spécifier un nom de conteneur. Exemple: `!restart grafana`")
            return True
            
        await manager.send_to_client(websocket, f"🔄 Tentative de redémarrage de '{arg}'...")
        try:
            # Find container by name or partial match
            containers = docker_client.containers.list(all=True)
            target_container = None
            for container in containers:
                if container.name.lower() == arg.lower() or arg.lower() in container.name.lower():
                    target_container = container
                    break
                    
            if not target_container:
                await manager.send_to_client(websocket, f"❌ Conteneur '{arg}' non trouvé.")
                return True
                
            target_container.restart()
            await manager.send_to_client(websocket, f"✅ Conteneur '{target_container.name}' redémarré avec succès.")
        except Exception as e:
            await manager.send_to_client(websocket, f"❌ Erreur lors du redémarrage: {str(e)}")
        return True
        
    elif cmd == "status":
        if not arg:
            # Show status of all containers
            await manager.send_to_client(websocket, "📊 Récupération du statut de tous les conteneurs...")
            try:
                containers = docker_client.containers.list(all=True)
                status_text = "📊 **Statut des conteneurs:**\n\n"
                for container in containers:
                    status = container.status
                    emoji = "🟢" if status == "running" else "🔴"
                    status_text += f"{emoji} **{container.name}**: {status}\n"
                await manager.send_to_client(websocket, status_text)
            except Exception as e:
                await manager.send_to_client(websocket, f"❌ Erreur lors de la récupération des statuts: {str(e)}")
        else:
            # Show status of specific container
            await manager.send_to_client(websocket, f"📊 Récupération du statut de '{arg}'...")
            try:
                containers = docker_client.containers.list(all=True)
                target_container = None
                for container in containers:
                    if container.name.lower() == arg.lower() or arg.lower() in container.name.lower():
                        target_container = container
                        break
                        
                if not target_container:
                    await manager.send_to_client(websocket, f"❌ Conteneur '{arg}' non trouvé.")
                    return True
                    
                status = target_container.status
                emoji = "🟢" if status == "running" else "🔴"
                
                # Get container stats if running
                stats_text = ""
                if status == "running":
                    try:
                        stats = target_container.stats(stream=False)
                        # Calculate CPU and memory usage
                        cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - stats["precpu_stats"]["cpu_usage"]["total_usage"]
                        system_delta = stats["cpu_stats"].get("system_cpu_usage", 0) - stats["precpu_stats"].get("system_cpu_usage", 0)
                        cpu_pct = 0.0
                        if system_delta > 0:
                            cpu_pct = (cpu_delta / system_delta * stats["cpu_stats"].get("online_cpus", 1) * 100.0)
                            
                        mem_usage = stats["memory_stats"].get("usage", 0)
                        mem_limit = stats["memory_stats"].get("limit", 1)
                        mem_pct = mem_usage / mem_limit * 100.0 if mem_limit else 0.0
                        
                        stats_text = f"\n- CPU: {cpu_pct:.2f}%\n- Mémoire: {mem_pct:.2f}% ({mem_usage / (1024*1024):.2f} MB / {mem_limit / (1024*1024):.2f} MB)"
                    except Exception:
                        stats_text = "\n- Statistiques non disponibles"
                
                status_text = f"{emoji} **{target_container.name}**: {status}{stats_text}"
                await manager.send_to_client(websocket, status_text)
            except Exception as e:
                await manager.send_to_client(websocket, f"❌ Erreur lors de la récupération du statut: {str(e)}")
        return True
    
    # Command not recognized
    return False


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
            
            # Handle special commands
            if await handle_command(message, websocket):
                continue
                
            # Détecter si c'est une simple salutation
            salutations = ["bonjour", "salut", "hello", "coucou", "hey", "bonsoir"]
            is_greeting = message.lower().strip().rstrip('!?.,;:') in salutations
            
            # Ne pas afficher "Je réfléchis" pour les salutations simples
            if not is_greeting:
                await manager.send_to_client(websocket, "🧠 Je réfléchis à votre question...")
            
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
    except Exception as e:
        print(f"Error in websocket connection: {str(e)}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
