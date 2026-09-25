# DAG для pipeline получения батч-предсказаний модели

from airflow import DAG   
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime 
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import numpy as np

from evidently import Report
from evidently.presets import DataDriftPreset


BASE_PATH = Path("/opt/airflow")  # вернет /opt/airflow
DATA_PATH = BASE_PATH / "data"    
MODEL_PATH = BASE_PATH / "models" 


def load_data():
    """
    Task 1: Загрузка сырых данных
    """

    df = pd.read_csv(DATA_PATH / "input/data.csv")

    # Сохраняем для следующего этапа пайплайна
    df.to_csv(DATA_PATH / "output/raw.csv", index=False)


def preprocess():
    """
    Task 2: Предобработка данных
    """

    # Читаем сырые данные с предыдущего шага
    df = pd.read_csv(DATA_PATH / "output/raw.csv")

    # 1. Преобразование order_datetime в формат pandas datetime
    df['order_datetime'] = pd.to_datetime(df['order_datetime'])

    # 2. Переводим traffic_level в формат строки
    df['traffic_level'] = df['traffic_level'].apply(lambda x: str(x))

    # 3. Выделение месяца, часа и дня недели из даты заказа
    df['month'] = df['order_datetime'].dt.month
    df['hour'] = df['order_datetime'].dt.hour

    # 4. Добавление нового признака: идёт ли дождь?
    df['is_rain'] = np.where(df['precip_mm'] > 11, 1, 0)

    # 5. Добавление признака дня недели
    df['dayofweek'] = df['order_datetime'].dt.dayofweek

    # 6. Заполняем пропущенные значения осадков и скорости ветра используя те же данные, что и во время обучения модели
    weather_data = pd.read_csv(MODEL_PATH / "preprocess_data/weather_data.csv", index_col=0)
    df = df.merge(weather_data, on=['restaurant_city', 'month'], how='left')
    df['precip_mm'] = df['precip_mm'].fillna(df['precip_mm_avg'])
    df['wind_speed'] = df['wind_speed'].fillna(df['wind_speed_avg'])
    df = df.drop(['precip_mm_avg', 'wind_speed_avg'], axis=1)

    # 7. Заполнение пропусков признака traffic_level - мода по часу заказа, дню недели и городу
    traffic_level = pd.read_csv(MODEL_PATH / "preprocess_data/traffic_level_data.csv", index_col=0)
    df = df.merge(traffic_level, on=['restaurant_city', 'hour', 'dayofweek'], how='left')
    df['traffic_level'] = df['traffic_level'].fillna(df['traffic_level_mode'])
    df = df.drop(['traffic_level_mode'], axis=1)

    
    # 8. Добавление нового признака: сколько заказов в этот час и день недели в среднем бывает
    avg_count_orders = pd.read_csv(MODEL_PATH / "preprocess_data/avg_count_orders.csv", index_col=0)

    df = df.merge(avg_count_orders, on=['restaurant_city','hour','dayofweek'], how='left')

    # 9. Сохраняем предобработанные данные для следующего этапа
    df.to_csv(DATA_PATH / "output/preprocessed.csv", index=False)


def prediction():
    """
    TASK 3: Выполнение предсказаний (инференс)
    Применяет обученную модель к предобработанным данным.
    """
    # Читаем предобработанные данные
    df = pd.read_csv(DATA_PATH / "output/preprocessed.csv")
    df['traffic_level'] = df['traffic_level'].apply(lambda x: str(x))

    for col in ['is_express_delivery', 'is_fast_food', 'is_rain']: 
        df[col] = df[col].apply(lambda x: bool(x))
    
    # Загружаем обученную модель
    model = joblib.load(MODEL_PATH / "catboost_model.joblib")

    # Выполняем предсказания
    df['prediction'] = model.predict(df.loc[:, ~df.columns.isin(['order_id', 'order_datetime'])])
    df.to_csv(DATA_PATH / "output/prediction_data.csv", index=False)


def data_drift():
    reference_data = pd.read_csv(DATA_PATH / "input/data_benchmark.csv")
    reference_data = reference_data.loc[:, ~reference_data.columns.isin(['order_id', 'order_datetime'])]

    df = pd.read_csv(DATA_PATH / "output/prediction_data.csv")
    current_data = df.loc[:, ~df.columns.isin(['order_id', 'order_datetime'])]

    report = Report([
    DataDriftPreset()
     ])

    data_drift_eval = report.run(
    current_data=current_data,
    reference_data=reference_data
    )

    # Считаем количество колонок, в которых наблюдался сдвиг данных
    data_drift_test = {}

    for metric in data_drift_eval.dict()['metrics']:
        if 'column' in metric['config']:
            data_drift_test[metric['config']['column']] = metric['value'] > 0.1

    count_data_drift_columns = sum(1 for v in data_drift_test.values() if v == True)

    # список колонок с флагом, обнаружился ли дрейф данных или нет, будем сохранять в датасет
    df['data_drift_columns'] = ', '.join([key for key in data_drift_test if data_drift_test[key] == True])
    df['count_data_drift_columns'] = count_data_drift_columns

    # Добавляем в столбец warning предупреждение, если был выявлен data drift
    if count_data_drift_columns > 0: 
        df['warning'] = 'prediction is NOT reliable'
    if count_data_drift_columns == 0:
        df['warning'] = 'ok'

    df.to_csv(DATA_PATH / "output/check_data_drift_prediction.csv", index=False)


def load_predictions_to_db():
    """
    Записываем предсказания в базу данных
    """

    df = pd.read_csv(DATA_PATH / "output/check_data_drift_prediction.csv")


    pg_hook = PostgresHook(postgres_conn_id="my_postgres-db-conn")

    rows_to_insert = list(df.itertuples(index=False, name=None))

    pg_hook.insert_rows(
                table='model_predictions',
                rows=rows_to_insert,
                target_fields=list(df.columns),
                commit_every=500,          # Коммитим каждые 500 строк
                replace=False              # False = INSERT, True = REPLACE
            )
            



with DAG(
    dag_id="prediction_time_delivery",
    start_date=datetime(2026, 8, 8),
    schedule=None,                         # Запуск каждый час: 0 * * * * 
    catchup=False,                         # Шедулер создаст и запустит DAG-раны для всех дат от start_date до текущего момента. 
                                           # Если catchup=False, Airflow запустит только свежий интервал, пропустив прошлые периоды
) as dag:

    load_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess
    )

    prediction_task = PythonOperator(
            task_id = 'prediction',
            python_callable=prediction
        )

    check_data_drift = PythonOperator(
        task_id = 'check_data_drift',
        python_callable = data_drift
    )

    

    fill_db_task = PythonOperator(
        task_id = "load_prediction_to_db",
        python_callable=load_predictions_to_db
    )

load_task >> preprocess_task >> prediction_task  >> check_data_drift >> fill_db_task

    

