FROM python:3.11-slim

# Arguments for caching layer when requirements change
ARG PIP_NO_CACHE_DIR=yes
WORKDIR /app

# Install runtime dependencies and Docker CLI
RUN apt-get update && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir requests pyyaml

# Copy source code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Default environment
ENV LOG_DIR=/app/logs \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

ENTRYPOINT ["python", "agent.py"]
