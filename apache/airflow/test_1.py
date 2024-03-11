from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': 60,
}

with DAG(
    'my_dag',
    default_args=default_args,
    description='A simple DAG',
    schedule_interval='@daily',
) as dag:

    task1 = DummyOperator(task_id='task1', dag=dag)
    task2 = DummyOperator(task_id='task2', dag=dag)
    task3 = BashOperator(
        task_id='bash_task',
        bash_command='echo "Hello, world"',
        dag=dag,
    )

    task1 >> task2
    task1 >> task3