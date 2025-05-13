"""
DAG to orchestrate the road accident data processing pipeline
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from docker.types import Mount

# Default arguments definition
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG creation
dag = DAG(
    'road_accidents',
    default_args=default_args,
    description='Road accident data processing pipeline',
    schedule_interval='@daily',
    start_date=datetime(2025, 5, 12),
    catchup=False,
    tags=['road_accidents', 'mlops'],
)

# Common mounts definition
common_mounts = [
    Mount(source='/home/ubuntu/mlops-road-accidents/data', target='/app/data', type='bind'),
    Mount(source='/home/ubuntu/mlops-road-accidents/config.yaml', target='/app/config.yaml', type='bind'),
]

# Task 1: Data extraction
extract_data = DockerOperator(
    task_id='extract_data',
    image='mlops-road-accidents_extract_data',
    command='python extract_data.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-road-accidents_default',
    mounts=common_mounts,
    mount_tmp_dir=False,  # Disable temporary mounting that causes problems
    auto_remove=True,
    dag=dag,
)

# Task 2: Synthetic data generation
synthet_data = DockerOperator(
    task_id='synthet_data',
    image='mlops-road-accidents_synthet_data',
    command='python synthet_data.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-road-accidents_default',
    mounts=common_mounts,
    mount_tmp_dir=False,  # Disable temporary mounting that causes problems
    auto_remove=True,
    dag=dag,
)

# Task 3: Data preparation
prepare_data = DockerOperator(
    task_id='prepare_data',
    image='mlops-road-accidents_prepare_data',
    command='python prepare_data.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-road-accidents_default',
    mounts=common_mounts,
    mount_tmp_dir=False,  # Disable temporary mounting that causes problems
    auto_remove=True,
    dag=dag,
)

# Task 4: Model training
train_model = DockerOperator(
    task_id='train_model',
    image='mlops-road-accidents_train_model',
    command='sh -c "rm -f /app/models/train_model.done && python train_model.py"',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-road-accidents_default',
    mounts=common_mounts + [
        Mount(source='/home/ubuntu/mlops-road-accidents/models', target='/app/models', type='bind'),
        Mount(source='/home/ubuntu/mlops-road-accidents/.git', target='/app/.git', type='bind'),
        Mount(source='/home/ubuntu/mlops-road-accidents/dvc.yaml', target='/app/dvc.yaml', type='bind'),
    ],
    mount_tmp_dir=False,  # Disable temporary mounting that causes problems
    auto_remove=True,
    dag=dag,
)

# Task 5 (save_best_model) has been removed

# Task 6: Data import
# Add a task to start PostgreSQL if needed
start_postgres = BashOperator(
    task_id='start_postgres',
    bash_command='docker start postgres_service || echo "Could not start existing container"',
    dag=dag,
)

# Modification of the import_data task to use the PostgreSQL service correctly
# No additional task is needed

# Modify the import_data task to use the host IP address instead of the container name
import_data = DockerOperator(
    task_id='import_data',
    image='mlops-road-accidents_import_data',
    command='sh -c "sleep 15 && python3 /app/import_data.py"',  # Increase the delay to ensure PostgreSQL is ready
    docker_url='unix://var/run/docker.sock',
    network_mode='host',  # Use host network to directly access PostgreSQL
    mounts=common_mounts + [
        Mount(source='/home/ubuntu/mlops-road-accidents/models', target='/app/models', type='bind'),
    ],
    environment={
        'POSTGRES_HOST': 'localhost',
        'POSTGRES_PORT': '5432',
        'POSTGRES_USER': 'postgres',
        'POSTGRES_PASSWORD': 'postgres',
        'POSTGRES_DB': 'road_accidents',
    },
    mount_tmp_dir=False,  # Disable temporary mounting that causes problems
    auto_remove=True,
    dag=dag,
)

# Dependencies definition
extract_data >> synthet_data >> prepare_data >> train_model >> start_postgres >> import_data
