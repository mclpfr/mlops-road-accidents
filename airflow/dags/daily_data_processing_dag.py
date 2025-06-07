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
    Source paths refer to paths accessible from the Airflow worker.
    If /opt/project is a mount from the host, these correspond to host paths.
    """
    print("Starting copy_current_data_for_evidently_task...")
    base_project_path_in_container = "/opt/project"
    source_file_relative_path = "data/processed/prepared_accidents_2023.csv"
    target_dir_relative_path = "evidently/current/"

    source_file_in_container = os.path.join(base_project_path_in_container, source_file_relative_path)
    target_dir_in_container = os.path.join(base_project_path_in_container, target_dir_relative_path)
    
    print(f"Source file path (in container): {source_file_in_container}")
    print(f"Target directory path (in container): {target_dir_in_container}")

    try:
        print(f"Attempting to create directory {target_dir_in_container} if it doesn't exist.")
        os.makedirs(target_dir_in_container, exist_ok=True)
        print(f"Ensured directory exists: {target_dir_in_container}")
    except Exception as e_mkdir:
        print(f"Error creating directory {target_dir_in_container}: {e_mkdir}")
        # Log user info if possible
        try:
            import pwd
            user_info = pwd.getpwuid(os.geteuid())
            print(f"Running as user: {user_info.pw_name} (UID: {os.geteuid()}, GID: {os.getegid()})")
        except Exception as e_user:
            print(f"Could not get user info: {e_user}")
        raise

    # Check if the target directory is writable
    if not os.access(target_dir_in_container, os.W_OK):
        print(f"Error: Target directory {target_dir_in_container} is not writable by the current user.")
        try:
            import pwd
            user_info = pwd.getpwuid(os.geteuid())
            print(f"Running as user: {user_info.pw_name} (UID: {os.geteuid()}, GID: {os.getegid()})")
            print(f"Permissions for {target_dir_in_container} (octal): {oct(os.stat(target_dir_in_container).st_mode)[-4:]}")
            parent_dir = os.path.dirname(target_dir_in_container.rstrip('/'))
            if parent_dir and os.path.exists(parent_dir):
                 print(f"Permissions for parent {parent_dir} (octal): {oct(os.stat(parent_dir).st_mode)[-4:]}")
        except Exception as e_diag:
            print(f"Could not get detailed diagnostic info: {e_diag}")
        raise PermissionError(f"Directory {target_dir_in_container} is not writable. Check logs for details.")
    else:
        print(f"Target directory {target_dir_in_container} is writable.")

    target_file_name = "current_data.csv"
    target_file_in_container = os.path.join(target_dir_in_container, target_file_name)
    print(f"Target file path (in container): {target_file_in_container}")
    
    if not os.path.exists(source_file_in_container):
        error_message = f"Source file {source_file_in_container} not found. Skipping copy for Evidently."
        print(error_message)
        raise FileNotFoundError(error_message)
    else:
        print(f"Source file {source_file_in_container} found.")

    try:
        print(f"Attempting to copy {source_file_in_container} to {target_file_in_container}")
        shutil.copy2(source_file_in_container, target_file_in_container)
        print(f"Successfully copied {source_file_in_container} to {target_file_in_container} for Evidently monitoring.")
    except Exception as e_copy:
        print(f"Error copying file for Evidently monitoring: {e_copy}")
        # Log user info again in case of error during copy, if not caught by os.access
        try:
            import pwd
            user_info = pwd.getpwuid(os.geteuid())
            print(f"Copy failed. Running as user: {user_info.pw_name} (UID: {os.geteuid()}, GID: {os.getegid()})")
        except Exception as e_user_copy:
            print(f"Could not get user info during copy failure: {e_user_copy}")
        raise

copy_current_data_for_evidently = PythonOperator(
    task_id='copy_current_data_for_evidently',
    python_callable=copy_current_data_for_evidently_task,
    dag=daily_processing_dag,
)
# --- End of Evidently Current Data Copy Task ---


# Dependencies definition for the daily processing DAG
extract_data_daily >> synthet_data_daily >> prepare_data_daily >> copy_current_data_for_evidently
