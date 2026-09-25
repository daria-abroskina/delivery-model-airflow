# DAG для однократного запуска при поднятии контейнера - создание таблицы

from airflow import DAG
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

# 1. Ручной запуск: создание таблицы
with DAG(
    dag_id='create_table',
    description='Create table with features and model predictions',
    schedule=None,
    catchup=False
) as dag:

    create_table = SQLExecuteQueryOperator(
        task_id = "create_prediction_table",
        conn_id="my_postgres-db-conn",
        sql="""
            CREATE TABLE IF NOT EXISTS model_predictions (
            order_id INTEGER,
            order_datetime TIMESTAMP,
            items_count INTEGER,
            distance_km FLOAT,
            is_express_delivery BOOL,
            vehicle_type VARCHAR(20), 
            base_speed_kmh FLOAT,
            restaurant_city VARCHAR(20),
            is_fast_food BOOL,
            prep_time_avg FLOAT,
            temperature FLOAT,
            precip_mm FLOAT,
            wind_speed FLOAT,
            traffic_level VARCHAR(5),
            month INTEGER,
            hour INTEGER,
            is_rain BOOL,
            dayofweek INTEGER,
            avg_orders FLOAT,
            prediction FLOAT,
            data_drift_columns VARCHAR(200),
            count_data_drift_columns INTEGER,
            warning VARCHAR(50),
            created_at TIMESTAMPTZ DEFAULT NOW()      
            )
            """
    )

create_table