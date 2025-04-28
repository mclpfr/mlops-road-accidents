FROM python:3.10-slim

# Arguments for UID and GID
ARG USER_ID=1000
ARG GROUP_ID=1000

WORKDIR /app

# Memory and environment config
ENV PYTHONUNBUFFERED=1
ENV PYTHONMEM=256m
ENV JAVA_OPTS="-Xmx512m -Xms256m"

COPY requirements.txt .
RUN pip install --upgrade pip && PIP_PROGRESS_BAR=off pip install --no-cache-dir -r requirements.txt \
    && apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/data /app/models && \
    groupadd -g ${GROUP_ID} appuser && \
    useradd -u ${USER_ID} -g appuser -s /bin/bash -m appuser && \
    chown -R appuser:appuser /app /app/data /app/models

COPY src/retrieve_data/. .

USER appuser

CMD ["python", "retrieve_data.py"]

