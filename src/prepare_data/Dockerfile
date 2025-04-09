FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt .

ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/data/processed

COPY config.yaml .

COPY src/prepare_data/prepare_data.py .

CMD ["python", "prepare_data.py"]
