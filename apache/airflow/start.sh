export AIRFLOW_HOME=`pwd`/../../tmp/apache-airflow

airflow tasks test example_bash_operator runme_0 2015-01-01

airflow dags backfill example_bash_operator \
    --start-date 2015-01-01 \
    --end-date 2015-01-02

airflow db migrate

airflow users create \
    --username admin \
    --firstname Peter \
    --lastname Parker \
    --role Admin \
    --email spiderman@superhero.org

airflow webserver --port 8080

airflow scheduler

