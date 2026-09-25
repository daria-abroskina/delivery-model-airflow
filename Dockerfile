FROM apache/airflow:3.1.6-python3.12

COPY ./requirements.txt ./requirements.txt 

RUN python -m pip install --upgrade pip && pip install -r requirements.txt 