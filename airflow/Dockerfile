FROM apache/airflow:2.7.1

USER root

# Install additional dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    default-libmysqlclient-dev \
    coreutils \
    login \
    docker.io \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create log directories with proper permissions
RUN mkdir -p /opt/airflow/logs /opt/airflow/logs/scheduler /opt/airflow/logs/webserver /opt/airflow/logs/worker && \
    chown 5000:5000 /opt/airflow/logs && \
    chmod -R 777 /opt/airflow/logs

# Add airflow user to the docker group to allow access to the Docker socket
RUN if ! getent group docker > /dev/null; then \
        groupadd docker; \
    fi && \
    usermod -aG docker airflow

# Copy scripts and set permissions
COPY --chown=5000:5000 airflow/scripts/ /opt/airflow/scripts/
RUN chmod +x /opt/airflow/scripts/entrypoint.sh && \
    ls -l /opt/airflow/scripts/entrypoint.sh

USER airflow

# Install Python packages
COPY airflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set entrypoint
ENTRYPOINT ["/opt/airflow/scripts/entrypoint.sh"]

WORKDIR /opt/airflow
