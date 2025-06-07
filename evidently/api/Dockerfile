FROM python:3.9-slim

# Arguments pour UID et GID
ARG USER_ID=1000
ARG GROUP_ID=1000

WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create appuser
RUN groupadd -g ${GROUP_ID} appuser && \
    useradd -u ${USER_ID} -g appuser -s /bin/bash -m appuser && \
    mkdir -p /app/reference /app/current && \
    chown -R ${USER_ID}:${GROUP_ID} /app

# Copy application code
COPY api.py api.py

# Set permissions
RUN chown -R ${USER_ID}:${GROUP_ID} /app

# Switch to non-root user
USER ${USER_ID}

# Expose the port the app runs on
EXPOSE 8001

# Commande pour lancer l'application
# Utilisation de --host 0.0.0.0 pour rendre le serveur accessible depuis l'extérieur du conteneur
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "debug"]
