"""
DAG for daily data extraction, synthesis, and preparation.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator # Added for the copy task
from docker.types import Mount
import shutil # Added for the copy task
import os     # Added for the copy task

# Default arguments definition
default_args_daily_processing = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 5, 12), # Adjust start_date as needed
}

# DAG creation for daily data processing
daily_processing_dag = DAG(
    'daily_data_processing', # New DAG ID
    default_args=default_args_daily_processing,
    description='Daily data extraction, synthesis, and preparation pipeline.',
    schedule_interval='@daily', # This DAG runs daily
    catchup=False,
    tags=['data_processing', 'daily', 'mlops'],
)

# Common mounts definition (same as in other DAGs)
common_mounts = [
    Mount(source='/home/ubuntu/mlops-road-accidents/data', target='/app/data', type='bind'),
    Mount(source='/home/ubuntu/mlops-road-accidents/config.yaml', target='/app/config.yaml', type='bind'),
]

# Task 1: Data extraction
extract_data_daily = DockerOperator(
    task_id='extract_data_daily',
    image='mlops-road-accidents_extract_data',
    command='python extract_data.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-road-accidents_default',
    mounts=common_mounts,
    mount_tmp_dir=False,
    auto_remove=True,
    dag=daily_processing_dag,
)

# Task 2: Synthetic data generation
synthet_data_daily = DockerOperator(
    task_id='synthet_data_daily',
    image='mlops-road-accidents_synthet_data',
    command='python synthet_data.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-road-accidents_default',
    mounts=common_mounts,
    mount_tmp_dir=False,
    auto_remove=True,
    dag=daily_processing_dag,
)

# Task 3: Data preparation
prepare_data_daily = DockerOperator(
    task_id='prepare_data_daily',
    image='mlops-road-accidents_prepare_data',
    command='python prepare_data.py',
    docker_url='unix://var/run/docker.sock',
    network_mode='mlops-road-accidents_default',
    mounts=common_mounts,
    mount_tmp_dir=False,
    auto_remove=True,
    dag=daily_processing_dag,
)

# --- Function and Task for Evidently Current Data Copy ---
def copy_current_data_for_evidently_task(**kwargs):
    """
    Copies the daily processed data to the 'current' directory for Evidently monitoring.
    Source: /home/ubuntu/mlops-road-accidents/data/processed/prepared_accidents_2023.csv (on host)
    Destination: /home/ubuntu/mlops-road-accidents/evidently/current/current_data_YYYYMMDD.csv (on host)
    """
    # Paths inside the container, assuming the project root is mounted to /opt/project
    container_project_root = "/opt/project"
    host_source_file_path = os.path.join(container_project_root, "data/processed/prepared_accidents_2023.csv")
    target_dir_on_host = os.path.join(container_project_root, "evidently/current/")
    
    os.makedirs(target_dir_on_host, exist_ok=True)
    
    target_file_name = "current_data.csv"
    target_file_path_on_host = os.path.join(target_dir_on_host, target_file_name)
    
    if not os.path.exists(host_source_file_path):
        error_message = f"Source file {host_source_file_path} not found. Skipping copy for Evidently."
        print(error_message)
        # Consider using AirflowSkipException if you want the task to be skipped rather than fail
        # from airflow.exceptions import AirflowSkipException
        # raise AirflowSkipException(error_message)
        raise FileNotFoundError(error_message) 

    try:
        shutil.copy2(host_source_file_path, target_file_path_on_host)
        print(f"Successfully copied {host_source_file_path} to {target_file_path_on_host} for Evidently monitoring.")
    except Exception as e:
        print(f"Error copying file for Evidently monitoring: {e}")
        raise

copy_current_data_for_evidently = PythonOperator(
    task_id='copy_current_data_for_evidently',
    python_callable=copy_current_data_for_evidently_task,
    dag=daily_processing_dag,
)
# --- End of Evidently Current Data Copy Task ---

# Dependencies definition for the daily processing DAG
extract_data_daily >> synthet_data_daily >> prepare_data_daily >> copy_current_data_for_evidently
