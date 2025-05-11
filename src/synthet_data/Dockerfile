FROM python:3.10-slim-buster

ARG USER_ID=1000
ARG GROUP_ID=1000

# Environment variables to optimize performance
ENV OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONMEM=384m \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Installing dependencies with cache cleanup in a single step
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pandas numpy pyyaml && \
    find /usr/local -name '*.pyc' -delete && \
    find /usr/local -name '__pycache__' -delete && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /app/data/raw && \
    groupadd -g ${GROUP_ID} appuser && \
    useradd -u ${USER_ID} -g appuser -s /bin/bash -m appuser && \
    chown -R appuser:appuser /app /app/data

COPY config.yaml .
COPY src/synthet_data/synthet_data.py .

USER appuser

CMD ["python", "synthet_data.py"] 
