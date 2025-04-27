FROM python:3.10-slim-buster

# Arguments pour UID et GID
ARG USER_ID=1000
ARG GROUP_ID=1000

WORKDIR /app

COPY requirements.txt .

# Configuration des limites de mémoire et d'optimisation
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONMEM=384m

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/data/processed /app/models && \
    # Créer un groupe et un utilisateur avec l'UID et GID spécifiés
    groupadd -g ${GROUP_ID} appuser && \
    useradd -u ${USER_ID} -g appuser -s /bin/bash -m appuser && \
    # Donner les permissions à l'utilisateur
    chown -R appuser:appuser /app /app/data /app/models

COPY config.yaml .

COPY src/prepare_data/prepare_data.py .

# Changer l'utilisateur pour les opérations suivantes
USER appuser

CMD ["python", "prepare_data.py"]
