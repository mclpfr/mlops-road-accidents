from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from docker.types import Mount
import shutil
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

# --- Python function to update Evidently 'reference' dataset ---
def update_reference_dataset_task(**kwargs):
    host_project_root = "/home/ubuntu/mlops-road-accidents"  # For logging clarity
    container_project_root = "/opt/project"

    source_relative_path = "data/processed/prepared_accidents_2023.csv"
    reference_relative_dir = "evidently/reference/"

    container_source_file_path = os.path.join(container_project_root, source_relative_path)
    container_reference_dir = os.path.join(container_project_root, reference_relative_dir)

    # Create directory using container path
    os.makedirs(container_reference_dir, exist_ok=True)

    target_file_name = "best_model_data.csv"
    
    container_target_file_path = os.path.join(container_reference_dir, target_file_name)
    
    # Check existence and copy using container paths
    if not os.path.exists(container_source_file_path):
        raise FileNotFoundError(f"Source file {container_source_file_path} (in container) not found for Evidently reference data.")
    
    shutil.copyfile(container_source_file_path, container_target_file_path)
    
    # For logging, show the host path so the user knows where to find it on their system
    host_target_file_path_for_log = os.path.join(host_project_root, reference_relative_dir, target_file_name)
    print(f"Successfully updated Evidently reference dataset. Host path: {host_target_file_path_for_log} (Container path: {container_target_file_path})")

# --- TASKS ---

# Task 1: Data extraction
extract_data = DockerOperator(
    task_id='extract_data', # Using standard name
    image='mlops-road-accidents_extract_data',
    command='python extract_data.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-road-accidents_default',
    mounts=common_mounts,
    mount_tmp_dir=False, auto_remove=True, dag=dag,
)

# Task 2: Synthetic data generation
synthet_data = DockerOperator(
    task_id='synthet_data', # Using standard name
    image='mlops-road-accidents_synthet_data',
    command='python synthet_data.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-road-accidents_default',
    mounts=common_mounts,
    mount_tmp_dir=False, auto_remove=True, dag=dag,
)

# Task 3: Data preparation
prepare_data = DockerOperator(
    task_id='prepare_data', # Using standard name
    image='mlops-road-accidents_prepare_data',
    command='python prepare_data.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-road-accidents_default',
    mounts=common_mounts,
    mount_tmp_dir=False, auto_remove=True, dag=dag,
)

# Task 5: Model training
train_model = DockerOperator(
    task_id='train_model',
    image='mlops-road-accidents_train_model',
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
    image='mlops-road-accidents_import_data',
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
update_evidently_reference = PythonOperator(
    task_id='update_evidently_reference_data',
    python_callable=update_reference_dataset_task,
    dag=dag,
)

# Dependencies definition
extract_data >> synthet_data >> prepare_data >> train_model >> start_postgres >> import_data >> update_evidently_reference
