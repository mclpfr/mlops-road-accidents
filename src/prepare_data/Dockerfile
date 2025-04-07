# Use a Python image with build-essential pre-installed
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file from the project root
COPY requirements.txt .

# Configure OpenBLAS to use a single thread
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

# Install Python dependencies with specific versions
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directory for processed data
RUN mkdir -p /app/data/processed

# Copy the configuration file
COPY config.yaml .

# Copy the specific script for this service
COPY src/prepare_data/prepare_data.py .

# The entry point will be defined in docker-compose
CMD ["python", "prepare_data.py"]
