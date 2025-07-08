# Streamlit microservice Dockerfile
# Use slim Python image for lightweight container
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libpq-dev \
    curl \
    docker.io \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project source will be mounted via docker-compose

# Expose Streamlit default port
EXPOSE 8501

# Command to run Streamlit app
CMD ["streamlit", "run", "streamlit/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
