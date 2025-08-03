import subprocess
from typing import Dict


def _run(cmd: str) -> str:
    """Execute *cmd* in a subshell and return stdout (stripped)."""
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    output = result.stdout.strip()
    if not output and result.stderr:
        output = f"[stderr]\n{result.stderr.strip()}"
    return output


def collect_docker_info(quick: bool = False) -> Dict[str, str]:
    """Collect docker diagnostics.

    Args:
        quick: If True, run only lightweight commands (stats & ps) to minimise latency.

    Returns:
        Dict[str, str] mapping section name -> output.
    """
    info: Dict[str, str] = {}

    # Always include basic stats & ps (already buffered to one-shot commands)
    info["docker_stats"] = _run("docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}'")
    info["docker_ps"] = _run("docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}'")

    if quick:
        return info  # skip heavy calls below

    # Additional (heavier) diagnostics – only when full mode is required
    # Show process lists for at most 3 containers (avoid huge output)
    info["docker_top"] = _run("docker ps -q | head -n 3 | xargs -r docker top")
    # Inspect at most 2 containers to reduce payload size
    info["docker_inspect"] = _run("docker ps -q | head -n 2 | xargs -r docker inspect --format '{{json .Config}}'")
    # Show the last 30 log lines from the first running container (if any)
    info["docker_logs"] = _run("docker logs --tail 30 $(docker ps -q | head -n 1)")

    return info


def _list_container_names() -> list[str]:
    """Return list of all container names (docker ps -a)."""
    names = _run('docker ps -a --format "{{.Names}}"')
    return names.splitlines() if names else []


def _find_container_by_partial(partial: str) -> str | None:
    """Return first container name that contains *partial* (case-insensitive)."""
    partial_l = partial.lower()
    for name in _list_container_names():
        if partial_l in name.strip().lower():
            return name.strip()
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

    # Tentative de résolution de nom partiel
    all_names = _list_container_names()
    resolved_name = None

    # 1. Recherche de correspondance exacte (insensible à la casse)
    for name in all_names:
        if name.lower() == container_name.lower():
            resolved_name = name
            break

    # 2. Si pas de correspondance exacte, recherche de correspondance partielle
    if not resolved_name:
        pmatch = _find_container_by_partial(container_name)
        if pmatch:
            resolved_name = pmatch

    # Si aucune correspondance trouvée, retourner une erreur informative
    if not resolved_name:
        return (
            f"Le conteneur '{container_name}' n'a pas été trouvé. "
            f"Conteneurs disponibles: {', '.join(all_names) or 'Aucun'}"
        )

    # Exécuter la commande logs
    result = subprocess.run(
        f"docker logs --tail {tail} {resolved_name}",
        shell=True,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        # Fallback: return stderr so that the caller can display it
        return result.stderr.strip()
    return result.stdout.strip() or "(Aucune sortie dans les logs)"
