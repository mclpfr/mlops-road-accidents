FROM python:3.10-slim-buster

# Arguments for UID and GID
ARG USER_ID=1000
ARG GROUP_ID=1000

WORKDIR /app

COPY requirements.txt .

# Memory limits and optimization configuration
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONMEM=384m

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/data/processed /app/models && \
    # Create group and user with specified UID and GID
    groupadd -g ${GROUP_ID} appuser && \
    useradd -u ${USER_ID} -g appuser -s /bin/bash -m appuser && \
    # Give permissions to the user
    chown -R appuser:appuser /app /app/data /app/models

COPY config.yaml .

COPY src/prepare_data/prepare_data.py .

# Switch user for subsequent operations
USER appuser

# Wait for extract_data.done file to be created before running prepare_data.py
CMD ["sh", "-c", "while [ ! -f /app/data/raw/extract_data.done ]; do echo 'Waiting for extract_data to finish...'; sleep 5; done; python prepare_data.py"]
