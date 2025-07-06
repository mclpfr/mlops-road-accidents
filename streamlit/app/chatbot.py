"""Streamlit page: Chatbot MLOps Agent (enhanced)

Fonctionnalités:
1. Exécuter toute commande `docker …` (et afficher la sortie)
2. Récupérer les logs d'un conteneur demandé
3. Sinon, collecter le contexte Docker et interroger le LLM Together.ai
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import streamlit as st
from io import BytesIO

def show_chatbot_page():
    """Displays the chatbot page."""
    # -----------------------------------------------------------------------------
    # Assurer que le package local `agent` est importé avant d'éventuels paquets tiers
    # -----------------------------------------------------------------------------
    REPO_ROOT = Path(__file__).resolve().parents[2]
    AGENT_DIR = REPO_ROOT / "agent"
    for p in (str(AGENT_DIR), str(REPO_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from docker_info_collector import collect_docker_info, get_container_logs
        from agent.agent import load_api_key, query_llm
    except ModuleNotFoundError as exc:
        st.error(f"Impossible d'importer les modules de l'agent : {exc}")
        st.stop()

    # -----------------------------------------------------------------------------
    # Interface utilisateur Streamlit
    # -----------------------------------------------------------------------------
    st.title("🤖 Agent MLOps")
    st.markdown(
        "Posez vos questions sur l'état de la plateforme Docker Compose (CPU, RAM, conteneurs, logs, erreurs, etc.)."
    )
    st.info(f"Using log collector from: `{get_container_logs.__code__.co_filename}`")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # [(role, content)]

    for role, content in st.session_state.chat_history:
        st.chat_message("user" if role == "user" else "assistant").markdown(content)

    user_input = st.chat_input("Votre question…")
    if not user_input:
        st.stop()

    st.session_state.chat_history.append(("user", user_input))
    st.chat_message("user").markdown(user_input)

    # -----------------------------------------------------------------------------
    # Réponses conviviales pour les salutations simples
    # -----------------------------------------------------------------------------
    # Salutations simples (bonjour…)
    if re.fullmatch(r"\s*(bonjour|salut|hello|hi|hey)[!\s]*", user_input, re.IGNORECASE):
        greeting_resp = "Bonjour ! 👋 Comment puis-je vous aider aujourd'hui ?"
        st.chat_message("assistant").markdown(greeting_resp)
        st.session_state.chat_history.append(("assistant", greeting_resp))
        st.stop()

    # Petit-bavardage : « ça va ? », « how are you ? »
    if re.search(r"\b(ca[ \-]?va\??|ça[ \-]?va\??|comment[ \-]?ç?a[ \-]?va\??|how are you\??)\b", user_input, re.IGNORECASE):
        smalltalk_resp = "Je vais très bien, merci ! 😊 Et vous, puis-je faire quelque chose pour vous ?"
        st.chat_message("assistant").markdown(smalltalk_resp)
        st.session_state.chat_history.append(("assistant", smalltalk_resp))
        st.stop()

    # -----------------------------------------------------------------------------
    # 1) Exécution directe d'une commande Docker
    # -----------------------------------------------------------------------------
    if re.match(r"^\s*(sudo\s+)?docker\s+.+", user_input, re.IGNORECASE):
        cmd = user_input.strip()
        if cmd.lower().startswith("sudo "):
            cmd = cmd[5:]
        with st.spinner(f"Exécution de : `{cmd}` …"):
            try:
                result = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=60)
                output = result.stdout.strip() or result.stderr.strip() or "(aucune sortie)"
            except Exception as exc:
                output = f"Erreur lors de l'exécution : {exc}"
        markdown_output = f"```bash\n{output}\n```"
        st.chat_message("assistant").markdown(markdown_output)
        st.session_state.chat_history.append(("assistant", markdown_output))
        st.stop()

    # -----------------------------------------------------------------------------
    # 2) Récupération des logs d'un conteneur
    # -----------------------------------------------------------------------------
    container = None
    # Priority 1: phrases like "logs de prometheus" / "logs du postgres"
    pattern_logs_de = r"logs?\s+(?:de\s+|du\s+|d['’]\s*)?(?P<name>[a-zA-Z0-9_.-]+)"
    match = re.search(pattern_logs_de, user_input, re.IGNORECASE)
    if match:
        container = match.group("name")
    else:
        # Priority 2: explicit docker command "docker logs prometheus"
        pattern_docker_logs = r"docker\s+logs\s+(?P<name>[a-zA-Z0-9_.-]+)"
        match = re.search(pattern_docker_logs, user_input, re.IGNORECASE)
        if match:
            container = match.group("name")
        else:
            # Priority 3: container name before the word logs: "prometheus logs"
            pattern_name_logs = r"(?P<name>[a-zA-Z0-9_.-]+)\s+logs?"
            match = re.search(pattern_name_logs, user_input, re.IGNORECASE)
            if match:
                # Guard against French articles like "les", "des" being captured
                candidate = match.group("name").lower()
                if candidate not in {"les", "des", "ses", "ces"}:
                    container = match.group("name")
    if container:
        with st.spinner(f"Récupération des logs du conteneur `{container}` …"):
            logs_output = get_container_logs(container, tail=200)
        markdown_logs = f"```\n{logs_output}\n```"
        st.chat_message("assistant").markdown(markdown_logs)
        st.session_state.chat_history.append(("assistant", markdown_logs))
        st.stop()
        with st.spinner(f"Récupération des logs du conteneur `{container}` …"):
            logs_output = get_container_logs(container, tail=200)
        markdown_logs = f"```\n{logs_output}\n```"
        st.chat_message("assistant").markdown(markdown_logs)
        st.session_state.chat_history.append(("assistant", markdown_logs))
        st.stop()

    # -----------------------------------------------------------------------------
    # 3) Fallback : analyse contextuelle via LLM
    # -----------------------------------------------------------------------------
    with st.spinner("Analyse en cours…"):
        try:
            context = "\n\n".join([f"### {k} ###\n{v}" for k, v in collect_docker_info().items()])[:15000]
            api_key = load_api_key()
            answer = query_llm(user_input, context, api_key)
        except Exception as exc:
            answer = f"⚠️ Erreur : {exc}"

    st.chat_message("assistant").markdown(answer)
    st.session_state.chat_history.append(("assistant", answer))