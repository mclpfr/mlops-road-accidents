FROM python:3.10-slim-buster

# Arguments pour UID et GID
ARG USER_ID=1000
ARG GROUP_ID=1000

WORKDIR /app

# Configuration de la limite de mémoire
ENV PYTHONUNBUFFERED=1
ENV PYTHONMEM=256m

RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/data /app/models && \
    groupadd -g ${GROUP_ID} appuser && \
    useradd -u ${USER_ID} -g appuser -s /bin/bash -m appuser && \
    chown -R appuser:appuser /app /app/data /app/models

COPY src/data_import/wait-for-postgres.sh .
RUN chmod +x wait-for-postgres.sh

COPY src/data_import/import_data.py .
COPY config.yaml .

USER appuser

CMD ["python", "import_data.py"] 
