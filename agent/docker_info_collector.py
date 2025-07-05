import subprocess
from typing import Dict


def _run(cmd: str) -> str:
    """Execute *cmd* in a subshell and return stdout (stripped)."""
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    output = result.stdout.strip()
    if not output and result.stderr:
        output = f"[stderr]\n{result.stderr.strip()}"
    return output


def collect_docker_info() -> Dict[str, str]:
    """Collect various docker diagnostics and return them in a dict."""
    info: Dict[str, str] = {}

    info["docker_stats"] = _run("docker stats --no-stream")
    info["docker_ps"] = _run("docker ps -a")
    # Show process lists for at most 5 containers (avoid huge output)
    info["docker_top"] = _run("docker ps -q | head -n 5 | xargs -r docker top")
    # Inspect at most 3 containers to reduce payload size
    info["docker_inspect"] = _run("docker ps -q | head -n 3 | xargs -r docker inspect")

    # Show the last 50 log lines from the first running container (if any)
    info["docker_logs"] = _run("docker logs --tail 50 $(docker ps -q | head -n 1)")

    return info


def _list_container_names() -> list[str]:
    """Return list of running container names (docker ps)."""
    names = _run("docker ps --format '{{{{.Names}}}}'")
    return names.splitlines() if names else []


def _find_container_by_partial(partial: str) -> str | None:
    """Return first container name that contains *partial* (case-insensitive)."""
    partial_l = partial.lower()
    for name in _list_container_names():
        if partial_l in name.lower():
            return name
    return None


def get_container_logs(container_name: str, tail: int = 100) -> str:
    """Return the last *tail* lines of logs for *container_name*.
    If the command fails, stderr is returned instead so the caller can display the error.
    """
    if not container_name:
        return "[erreur] Le nom du conteneur est vide."
    # Sanitize container name to avoid shell injection – allow alphanum, dash, underscore, dot
    import re
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", container_name):
        return "[error] Container name contains invalid characters."
    # Tenter de résoudre un nom partiel si le conteneur exact n'existe pas
    resolved_name = container_name
    test_rc = subprocess.run(
        f"docker inspect {container_name}",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    if test_rc != 0:
        pmatch = _find_container_by_partial(container_name)
        if pmatch:
            resolved_name = pmatch
    # Exécuter la commande logs
    result = subprocess.run(
        f"docker logs --tail {tail} {resolved_name}",
        shell=True,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        if "No such container" in result.stderr:
            return f"Le conteneur '{container_name}' n'existe pas ou n'est pas en cours d'exécution."
        # Fallback: return stderr so that the caller can display it
        return result.stderr.strip()
    return result.stdout.strip() or "(Aucune sortie dans les logs)"
