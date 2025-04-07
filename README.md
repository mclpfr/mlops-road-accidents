# Road Accident Prediction Project - MLOps

## Project Overview
This project aims to predict the severity of road accidents in France using machine learning techniques. The goal is to provide real-time urgency estimates for police and medical services.

## Prerequisites
- Python 3.10
- Python libraries: 
  - pandas
  - scikit-learn
  - numpy
  - requests
  - beautifulsoup4
  - pytest
  - fastapi
  - uvicorn
  - pyyaml
  - python-multipart

## Installation
1. Clone the repository
```bash
git clone https://github.com/mclpfr/mlops-road-accidents.git
```
2. Create a virtual environment
```bash
python3.10 -m venv venv
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Data Extraction
```bash
python src/extract_data.py
```

### Data Preprocessing
```bash
python src/prepare_data.py
```

### Model Training
```bash
python src/train_model.py
```

### Run all services
```bash
cd src
uvicorn api:app --reload
```

### Run unit tests
```bash
cd tests
pytest tests.py
```

## Usage with Docker Compose

### Run all services

To run all services defined in the `docker-compose.yml` file (respecting dependencies):

```bash
docker-compose up -d
```
To run a single microservice *without* starting its dependencies:

```bash
docker-compose up --no-deps prepare_data
```

To stop services:

```bash
docker-compose down
```
