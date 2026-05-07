# sales-forecasting-project

# End-to-End Time Series Forecasting System

## Objective
Forecast next 8 weeks of sales for each state using historical data.

## Models Implemented
- ARIMA/SARIMA
- Prophet
- XGBoost
- LSTM

## Features Used
- Lag Features
- Rolling Mean & Std
- Day of Week
- Month
- Holiday Flag

## Run Project

Install dependencies:

pip install -r requirements.txt

Run training:

python main.py

Run API:

cd api
uvicorn app:app --reload

Open API Docs:

http://127.0.0.1:8000/docs
