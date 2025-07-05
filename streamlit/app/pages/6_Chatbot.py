"""Streamlit page: Chatbot MLOps Agent"""
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# Ajouter un bouton de retour à l'accueil dans la barre latérale
if st.sidebar.button("← Retour à l'accueil"):
    st.switch_page("main.py")

# Add repo root to PYTHONPATH to import agent modules when running inside Docker
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from agent.docker_info_collector import (
        collect_docker_info,
        get_container_logs,
    )  # type: ignore
    from agent.agent import load_api_key, query_llm  # type: ignore
except ModuleNotFoundError as exc:
    st.error(f"Impossible d'importer les modules de l'agent : {exc}")
    st.stop()

st.title("🤖 Chatbot MLOps / Docker")

st.markdown(
    "Posez vos questions sur l'état de la plateforme Docker Compose (CPU, RAM, conteneurs, logs, erreurs, etc.)."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content)

# Display previous messages
for role, content in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(content)
    else:
        st.chat_message("assistant").markdown(content)

import re

user_input = st.chat_input("Votre question…")
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    st.chat_message("user").markdown(user_input)

    # Détection d'un simple salut
    if re.match(r"^\s*(bonjour|salut|hello)\s*$", user_input, re.IGNORECASE):
        answer = "Bonjour, que puis-je faire pour vous aider ?"
        st.session_state.chat_history.append(("assistant", answer))
        st.chat_message("assistant").markdown(answer)
        st.stop()

    # If the user explicitly asks for container logs, run the command directly
    matched = re.search(r"logs(?:\s+(?:du|de|d'))?\s+(?:conteneur\s+)?(?P<name>[A-Za-z0-9_.-]+)|docker\s+logs\s+(?P<name_dk>[A-Za-z0-9_.-]+)", user_input, re.IGNORECASE)
    if matched:
        container = matched.group("name") or matched.group("name_dk")
        with st.spinner(f"Récupération des logs du conteneur {container}…"):
            logs_output = get_container_logs(container, tail=200)
        markdown_logs = f"```\n{logs_output}\n```"
        st.session_state.chat_history.append(("assistant", markdown_logs))
        st.chat_message("assistant").markdown(markdown_logs)
        answer = None  # Already responded
    else:
        # Gather context and query LLM
        with st.spinner("Analyse en cours…"):
            try:
                context = "\n\n".join(
                    [f"### {k} ###\n{v}" for k, v in collect_docker_info().items()]
                )[:15000]
                api_key = load_api_key()
                answer = query_llm(user_input, context, api_key)
            except Exception as exc:
                answer = f"⚠️ Erreur : {exc}"

    if answer is not None:
        st.session_state.chat_history.append(("assistant", answer))
        st.chat_message("assistant").markdown(answer)
