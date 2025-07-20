FROM python:3.10-slim-bullseye

# Arguments pour UID et GID
ARG USER_ID=1000
ARG GROUP_ID=1000

WORKDIR /app

COPY requirements.txt .

# Install git for commit hash and metadata commit
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Configuration des limites de mémoire et d'optimisation
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONMEM=512m
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

RUN pip install --no-cache-dir -r requirements.txt

# Install DVC
RUN pip install --no-cache-dir dvc

RUN mkdir -p /app/models /app/data && \
    groupadd -g ${GROUP_ID} appuser && \
    useradd -u ${USER_ID} -g appuser -s /bin/bash -m appuser && \
    chown -R appuser:appuser /app /app/data /app/models

COPY config.yaml .

COPY src/train_model/train_model.py .

# Configure Git identity for commits
RUN git config --global user.email "marco@datascientest.com" && \
    git config --global user.name "Marco LOPES"

USER appuser

CMD ["python", "train_model.py"]
