from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load('../models/best_model.pkl')

FEATURES = [
    'lag_1',
    'lag_7',
    'lag_30',
    'rolling_mean_7',
    'rolling_std_7',
    'day_of_week',
    'month',
    'is_holiday'
]

@app.get('/')
def home():
    return {'message': 'Sales Forecasting API Running'}

@app.post('/predict')
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df[FEATURES])

    return {
        'forecast': prediction.tolist()
    }
