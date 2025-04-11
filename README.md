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
  - mlflow

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
python3.10 src/extract_data/extract_data.py
```

### Data Preprocessing
```bash
python3.10 src/prepare_data/prepare_data.py
```

### Model Training
```bash
python3.10 src/train_model/train_model.py
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

## Configuration

### MLflow Configuration

You can obtain your Dagshub token from your Dagshub account:
1. Go to https://dagshub.com/user/settings/tokens
2. Create a new token
3. Copy the token and use it as the value for MLFLOW_TRACKING_PASSWORD

### Configuration File

Create a local `config.yaml` file based on the template `config.yaml.example`:

```bash
cp config.yaml.example config.yaml
```

Then edit `config.yaml` with your personal information. Here are the variables you need to configure:

```yaml
data_extraction:
  year: "2023"  
  url: "https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/"

mlflow:
  tracking_uri: "https://dagshub.com/YOUR_USERNAME/mlops-road-accidents.mlflow" 
  username: "YOUR_USERNAME"  
  password: "YOUR_DAGSHUB_TOKEN"
```

Replace the following placeholders with your actual information:
- `YOUR_USERNAME`: Your Dagshub username
- `YOUR_DAGSHUB_TOKEN`: The API token you generated from Dagshub
