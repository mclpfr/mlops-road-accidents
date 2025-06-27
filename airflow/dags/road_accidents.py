from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from docker.types import Mount
import os

# Default arguments definition
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG creation - This DAG is comprehensive and triggered on demand
dag = DAG(
    'road_accidents', # User requested DAG ID
    default_args=default_args,
    description='Comprehensive road accident pipeline: data processing, training, import, and Evidently updates. Triggered on demand.',
    schedule_interval=None,  # No automatic daily schedule, triggered by alert or manually
    start_date=datetime(2025, 5, 12),
    catchup=False,
    tags=['mlops', 'road_accidents', 'on_demand_pipeline'],
)

# Common mounts definition
common_mounts = [
    Mount(source='/home/ubuntu/mlops-road-accidents/data', target='/app/data', type='bind'),
    Mount(source='/home/ubuntu/mlops-road-accidents/config.yaml', target='/app/config.yaml', type='bind'),
]



# --- TASKS ---

# Task 1: Data extraction
extract_data = DockerOperator(
    task_id='extract_data', # Using standard name
    image='mlops-road-accidents-extract_data',
    command='python extract_data.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-road-accidents_default',
    mounts=common_mounts,
    mount_tmp_dir=False, auto_remove=True, dag=dag,
)

# Task 2: Synthetic data generation
synthet_data = DockerOperator(
    task_id='synthet_data', # Using standard name
    image='mlops-road-accidents-synthet_data',
    command='python synthet_data.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-road-accidents_default',
    mounts=common_mounts,
    mount_tmp_dir=False, auto_remove=True, dag=dag,
)

# Task 3: Data preparation
prepare_data = DockerOperator(
    task_id='prepare_data', # Using standard name
    image='mlops-road-accidents-prepare_data',
    command='python prepare_data.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-road-accidents_default',
    mounts=common_mounts,
    mount_tmp_dir=False, auto_remove=True, dag=dag,
)

# Task 5: Model training
train_model = DockerOperator(
    task_id='train_model',
    image='mlops-road-accidents-train_model',
    command='sh -c "rm -f /app/models/train_model.done && python train_model.py"',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-road-accidents_default',
    mounts=common_mounts + [
        Mount(source='/home/ubuntu/mlops-road-accidents/models', target='/app/models', type='bind'),
        Mount(source='/home/ubuntu/mlops-road-accidents/data/processed', target='/app/data/processed', type='bind')
    ],
    mount_tmp_dir=False, auto_remove=True, dag=dag,
)

# Task 6: Start PostgreSQL
start_postgres = BashOperator(
    task_id='start_postgres',
    bash_command='docker start postgres_service || echo "Could not start existing container"',
    dag=dag,
)

# Task 7: Import data
import_data = DockerOperator(
    task_id='import_data',
    image='mlops-road-accidents-import_data',
    command='python import_data.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-road-accidents_default',
    mounts=common_mounts + [
        Mount(source='/home/ubuntu/mlops-road-accidents/data/processed', target='/app/data/processed', type='bind'),
        Mount(source='/home/ubuntu/mlops-road-accidents/models', target='/app/models', type='bind')
    ],
    mount_tmp_dir=False, auto_remove=True, dag=dag,
)

# Task 8: Update Evidently 'reference' data
update_evidently_reference = DockerOperator(
    task_id='update_evidently_reference_data',
    image='alpine:latest',
    command=[
        "sh", "-c",
        "mkdir -p /opt/project/evidently/reference/ && "
        "cp /opt/project/data/processed/prepared_accidents_2023.csv /opt/project/evidently/reference/best_model_data.csv"
    ],
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-road-accidents_default',
    mounts=[
        Mount(source='/home/ubuntu/mlops-road-accidents', target='/opt/project', type='bind')
    ],
    mount_tmp_dir=False,
    auto_remove=True,
    dag=dag,
    user='root', # Run as root to ensure permissions
)

# Dependencies definition
extract_data >> synthet_data >> prepare_data >> train_model >> start_postgres >> import_data >> update_evidently_reference
