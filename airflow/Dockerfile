FROM apache/airflow:2.7.1

USER root

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN if getent group $GROUP_ID >/dev/null 2>&1; then \
        echo "Group $GROUP_ID already exists, skipping creation."; \
    else \
        groupadd -g $GROUP_ID airflow || echo "Failed to create group, continuing..."; \
    fi && \
    if id -u $USER_ID >/dev/null 2>&1; then \
        echo "User $USER_ID already exists, skipping creation."; \
    else \
        useradd --no-log-init -ms /bin/bash -u $USER_ID -g $GROUP_ID airflow || echo "Failed to create user, continuing..."; \
    fi

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    default-libmysqlclient-dev \
    coreutils \
    login \
    docker.io \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/airflow/logs /opt/airflow/logs/scheduler /opt/airflow/logs/webserver /opt/airflow/logs/worker && \
    chown -R $USER_ID:$GROUP_ID /opt/airflow && \
    chmod -R 777 /opt/airflow/logs

RUN if ! getent group docker > /dev/null; then \
        groupadd docker; \
    fi && \
    usermod -aG docker airflow && usermod -aG $GROUP_ID airflow

RUN mkdir -p /opt/project && \
    chmod -R 777 /opt/project

COPY --chown=$USER_ID:$GROUP_ID airflow/airflow.cfg /opt/airflow/airflow.cfg
COPY --chown=$USER_ID:$GROUP_ID airflow/scripts/ /opt/airflow/scripts/
RUN chmod +x /opt/airflow/scripts/entrypoint.sh && \
    chmod +x /opt/airflow/scripts/load_config.py && \
    chmod 644 /opt/airflow/airflow.cfg

COPY airflow/requirements.txt /tmp/requirements.txt

USER airflow
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.1/constraints-3.8.txt"

# Création du répertoire pour les fichiers statiques
RUN mkdir -p /opt/airflow/www/static && \
    # Copie des fichiers statiques depuis le site-packages d'Airflow
    find /home/airflow/.local -name 'static' -type d -path '*/airflow/*' | xargs -I {} cp -r {}/. /opt/airflow/www/static/ || echo "Aucun fichier statique trouvé dans le site-packages d'Airflow"

ENTRYPOINT ["/opt/airflow/scripts/entrypoint.sh"]

WORKDIR /opt/airflow
