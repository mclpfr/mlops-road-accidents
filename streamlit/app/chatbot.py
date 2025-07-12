"""Streamlit page: Gérard - Agent MLOps Autonome

Cette page affiche une interface iframe qui se connecte au service agent
via WebSocket pour fournir une interface de chat en temps réel.
"""
from __future__ import annotations

import streamlit as st
import docker
from pathlib import Path

def show_chatbot_page():
    """Displays the chatbot page with an iframe to the agent interface."""
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
    
    # Déterminer l'URL de l'agent
    # URL interne pour les communications entre conteneurs
    agent_internal_url = "http://agent:8002"
    # URL externe pour l'accès depuis le navigateur
    agent_external_url = "http://localhost:8002"
    
    # Créer un iframe pour afficher l'interface de l'agent
    st.components.v1.iframe(
        src=agent_external_url,
        height=700,
        scrolling=True
    )
    
    # Ajouter un lien direct vers l'interface de l'agent en cas de problème avec l'iframe
    st.markdown(f"""Si l'iframe ne s'affiche pas correctement, [cliquez ici pour accéder directement à l'interface de Gérard]({agent_external_url}).""")

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
    st.markdown("""
    <div style="margin-top: 20px; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
        <p>Si l'iframe ne s'affiche pas correctement, vous pouvez accéder directement à l'interface de Gérard à l'adresse suivante :</p>
        <a href="http://localhost:8002" target="_blank">http://localhost:8002</a>
    </div>
    """, unsafe_allow_html=True)