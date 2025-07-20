FROM python:3.10-slim-bullseye

ARG USER_ID=1000
ARG GROUP_ID=1000

WORKDIR /app

COPY requirements.txt .

ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONMEM=256m

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/data/raw && \
    groupadd -g ${GROUP_ID} appuser && \
    useradd -u ${USER_ID} -g appuser -s /bin/bash -m appuser && \
    chown -R appuser:appuser /app /app/data

COPY config.yaml .
COPY src/extract_data/extract_data.py .

USER appuser

CMD ["python", "extract_data.py"]
