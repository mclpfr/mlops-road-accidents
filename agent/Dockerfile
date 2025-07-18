FROM python:3.11-slim

# Arguments for caching layer when requirements change
ARG PIP_NO_CACHE_DIR=yes
WORKDIR /app

# Install runtime dependencies and Docker CLI
RUN apt-get update && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir requests pyyaml docker fastapi uvicorn websockets

# Copy source code
COPY . .
COPY agent/services.yml /app/services.yml
COPY config.yaml /app/config.yaml

# Create directories for static files and logs
RUN mkdir -p /app/static /app/logs


# Default environment
ENV LOG_DIR=/app/logs \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Expose the FastAPI port
EXPOSE 8003

# Run both the monitoring daemon and the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]
