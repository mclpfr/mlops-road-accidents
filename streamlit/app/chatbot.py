"""Streamlit page: Gérard - Agent MLOps Autonome

Cette page affiche une interface iframe qui se connecte au service agent
via WebSocket pour fournir une interface de chat en temps réel.
"""
from __future__ import annotations

import streamlit as st
import os
import docker
import requests
from pathlib import Path

def show_chatbot_page():
    """Displays the chatbot page with an iframe to the agent interface."""
    # Get agent external URL from environment variable or use default
    agent_external_url = os.environ.get('AGENT_EXTERNAL_URL', 'https://srv877984.hstgr.cloud/agent/')
    
    # Determine agent URL based on environment
    is_docker = os.path.exists('/.dockerenv')
    # L'URL interne pour les requêtes de serveur à serveur dans Docker
    agent_internal_url = 'http://agent:8003' if is_docker else agent_external_url
    # L'URL externe pour l'iframe côté client
    agent_url_for_iframe = agent_external_url
    
    # -----------------------------------------------------------------------------
    # Configuration de l'interface
    # -----------------------------------------------------------------------------
    st.title("🤖 Gérard - Agent MLOps Autonome")
    
    # Afficher une description de l'agent
    st.markdown("""
    Gérard est votre assistant MLOps autonome. Il peut surveiller vos conteneurs Docker, 
    exécuter des commandes, et vous aider à diagnostiquer des problèmes dans votre infrastructure.
    
    ### Fonctionnalités
    - Surveillance en temps réel des conteneurs Docker
    - Affichage des logs des conteneurs
    - Redémarrage des services en cas de problème
    - Analyse des performances et des erreurs
    
    Interagissez directement avec Gérard dans l'interface ci-dessous.
    """)
    

    
    # Check if agent is reachable
    agent_available = False
    try:
        # Utiliser l'URL interne pour la vérification de santé
        response = requests.get(f"{agent_internal_url}/healthz", timeout=2)
        agent_available = response.status_code == 200
    except (requests.RequestException, ConnectionError):
        pass
    
    if not agent_available:
        st.warning("⚠️ L'agent MLOps n'est pas accessible. Veuillez démarrer le service 'agent'.")
        if st.button("Rafraîchir l'état"):
            st.rerun()
        return
    
    # Create WebSocket connection URL (ws:// for local, wss:// for production)
    ws_protocol = "ws" if "localhost" in agent_url_for_iframe or "127.0.0.1" in agent_url_for_iframe else "wss"
    ws_url = agent_url_for_iframe.replace("http", "ws").replace("https", "wss") + "/ws"
    
    # Create iframe with proper WebSocket URL
    iframe_html = f"""
    <style>
        .chat-container {{
            width: 100%;
            height: 700px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }}
        .chat-iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}
    </style>
    <div class="chat-container">
        <iframe 
            src="{agent_url_for_iframe}" 
            class="chat-iframe"
            sandbox="allow-forms allow-scripts allow-same-origin allow-popups"
            allow="microphone; camera; geolocation">
        </iframe>
    </div>
    <script>
        // Ensure WebSocket connection is properly established
        const socket = new WebSocket('{ws_url}');
        
        socket.onopen = function(e) {{
            console.log('[WebSocket] Connected to agent');
        }};
        
        socket.onmessage = function(event) {{
            console.log('[WebSocket] Message received:', event.data);
            // Handle incoming messages if needed
        }};
        
        socket.onclose = function(e) {{
            if (e && e.wasClean) {{
                const reason = e.reason || 'No reason provided';
                console.log('[WebSocket] Connection closed cleanly, code=' + e.code + ' reason=' + reason);
            }} else {{
                console.error('[WebSocket] Connection died');
                // Attempt to reconnect after 5 seconds
                setTimeout(function() {{ window.location.reload(); }}, 5000);
            }}
        }};
        
        socket.onerror = function(error) {{
            console.error(`[WebSocket] Error:`, error);
        }};
    </script>
    """
    st.components.v1.html(iframe_html, height=720, scrolling=False)
    
    # Ajouter un lien direct vers l'interface de l'agent en cas de problème avec l'iframe
    st.markdown(f"""Si l'iframe ne s'affiche pas correctement, [cliquez ici pour accéder directement à l'interface de Gérard]({agent_url_for_iframe}).""")

    # Ajouter un bouton pour rafraîchir l'iframe si nécessaire
    if st.button("Rafraîchir l'interface"):
        st.rerun()

    # -----------------------------------------------------------------------------
    # Agent enable / disable controls
    # -----------------------------------------------------------------------------
    try:
        cli = docker.from_env()
        agent_container = cli.containers.get("agent")
        is_running = agent_container.status == "running"
    except Exception:
        agent_container = None
        is_running = False

    status_str = "🟢 Actif" if is_running else "🔴 Inactif"
    st.markdown(f"**Statut de l'agent :** {status_str}")

    # Boutons pour activer/désactiver l'agent
    if agent_container:
        col1, col2 = st.columns(2)
        with col1:
            if is_running and st.button("🛑 Désactiver l'agent"):
                try:
                    agent_container.stop(timeout=10)
                    st.success("Agent arrêté.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Erreur : {exc}")
        with col2:
            if not is_running and st.button("▶️ Activer l'agent"):
                try:
                    agent_container.start()
                    st.success("Agent démarré.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Erreur : {exc}")
    else:
        st.warning("Conteneur 'agent' introuvable.")
    
    # -----------------------------------------------------------------------------
    # Informations sur l'utilisation
    # -----------------------------------------------------------------------------
    st.markdown("### Commandes utiles")
    
    st.markdown("""
    Vous pouvez utiliser les commandes suivantes dans l'interface de Gérard :
    
    - `!help` : Affiche l'aide des commandes disponibles
    - `!logs <conteneur>` : Affiche les logs d'un conteneur
    - `!restart <conteneur>` : Redémarre un conteneur
    - `!status [<conteneur>]` : Affiche le statut d'un ou de tous les conteneurs
    
    Vous pouvez également poser des questions en langage naturel comme :
    - "Est-ce que Grafana fonctionne bien ?"
    - "Quels conteneurs utilisent trop de mémoire ?"
    - "Y a-t-il des erreurs critiques dans les logs ?"
    """)
    
    # Ajouter un lien pour accéder directement à l'interface de l'agent
    st.markdown(f"""
    <div style="margin-top: 20px; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
        <p>Si l'iframe ne s'affiche pas correctement, vous pouvez accéder directement à l'interface de Gérard à l'adresse suivante :</p>
        <a href=\"{agent_url_for_iframe}\" target=\"_blank\">{agent_url_for_iframe}</a>
    </div>
    """, unsafe_allow_html=True)
