from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create a DAG instance
dag = DAG(
    'road_accidents',
    default_args=default_args,
    description='MLOps pipeline for road accidents data in Kubernetes',
    schedule_interval='@daily',
    catchup=False,
)

# Define the namespace
namespace = 'mlops-road-accidents'

# Task 1: Extract Data
extract_data = BashOperator(
    task_id='extract_data',
    bash_command=f"kubectl run extract-data-{{{{ ts_nodash }}}} --namespace={namespace} --image=mclpfr/extract_data_service:latest --restart=Never --rm --attach",
    dag=dag,
)

# Task 2: Synthet Data
synthet_data = BashOperator(
    task_id='synthet_data',
    bash_command=f"kubectl run synthet-data-{{{{ ts_nodash }}}} --namespace={namespace} --image=mclpfr/synthet_data_service:latest --restart=Never --rm --attach",
    dag=dag,
)

# Task 3: Prepare Data
prepare_data = BashOperator(
    task_id='prepare_data',
    bash_command=f"kubectl run prepare-data-{{{{ ts_nodash }}}} --namespace={namespace} --image=mclpfr/prepare_data_service:latest --restart=Never --rm --attach",
    dag=dag,
)

# Task 4: Train Model
train_model = BashOperator(
    task_id='train_model',
    bash_command=f"kubectl run train-model-{{{{ ts_nodash }}}} --namespace={namespace} --image=mclpfr/train_model_service:latest --restart=Never --rm --attach",
    dag=dag,
)

# Task 5: Import Data
import_data = BashOperator(
    task_id='import_data',
    bash_command=f"kubectl run import-data-{{{{ ts_nodash }}}} --namespace={namespace} --image=mclpfr/import_data_service:latest --restart=Never --rm --attach",
    dag=dag,
)

# Define the task dependencies
extract_data >> synthet_data >> prepare_data >> train_model >> import_data
