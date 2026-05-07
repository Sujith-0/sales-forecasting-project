import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor

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

def train_models(df):

    split_date = df['Date'].max() - pd.Timedelta(days=56)

    train = df[df['Date'] < split_date]

    val = df[df['Date'] >= split_date]

    results = {}

    xgb = XGBRegressor()

    xgb.fit(train[FEATURES], train['Total'])

    xgb_preds = xgb.predict(val[FEATURES])

    xgb_mae = mean_absolute_error(val['Total'], xgb_preds)

    results['XGBoost'] = xgb_mae

    try:

        state_df = train[train['State'] == train['State'].iloc[0]]

        model = SARIMAX(
            state_df['Total'],
            order=(1,1,1),
            seasonal_order=(1,1,1,7)
        )

        arima_result = model.fit(disp=False)

        arima_forecast = arima_result.forecast(steps=len(val))

        arima_mae = mean_absolute_error(
            val['Total'].values[:len(arima_forecast)],
            arima_forecast
        )

        results['ARIMA'] = arima_mae

    except:

        results['ARIMA'] = 999999

    try:

        prophet_train = train[['Date', 'Total']]

        prophet_train = prophet_train.rename(
            columns={
                'Date': 'ds',
                'Total': 'y'
            }
        )

        prophet_model = Prophet()

        prophet_model.fit(prophet_train)

        future = prophet_model.make_future_dataframe(periods=56)

        forecast = prophet_model.predict(future)

        prophet_preds = forecast['yhat'].tail(56).values

        prophet_mae = mean_absolute_error(
            val['Total'].values[:56],
            prophet_preds
        )

        results['Prophet'] = prophet_mae

    except:

        results['Prophet'] = 999999

    best_model_name = min(results, key=results.get)

    print(results)

    print("Best Model:", best_model_name)

    if best_model_name == 'XGBoost':

        joblib.dump(xgb, 'models/best_model.pkl')

    return results