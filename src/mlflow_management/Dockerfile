FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY untag_best_model.py .

ENV PYTHONPATH "${PYTHONPATH}:/opt/project"

VOLUME /opt/project

CMD ["python", "untag_best_model.py"]
