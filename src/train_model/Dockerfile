FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt .

ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models

COPY config.yaml .

COPY src/train_model/train_model.py .

CMD ["python", "train_model.py"]
