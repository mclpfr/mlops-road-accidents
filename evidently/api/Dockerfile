FROM python:3.9-slim

ARG USER_ID=1000
ARG GROUP_ID=1000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN groupadd -g ${GROUP_ID} appuser && \
    useradd -u ${USER_ID} -g appuser -s /bin/bash -m appuser && \
    mkdir -p /app/reference /app/current /app/static && \
    chown -R ${USER_ID}:${GROUP_ID} /app

COPY config.yaml config.yaml
COPY api.py api.py
COPY static/ /app/static/

RUN chown -R ${USER_ID}:${GROUP_ID} /app

USER ${USER_ID}

EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "debug", "--reload"]
