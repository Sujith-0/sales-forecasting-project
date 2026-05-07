import holidays

def create_features(df):

    df['lag_1'] = df.groupby('State')['Total'].shift(1)

    df['lag_7'] = df.groupby('State')['Total'].shift(7)

    df['lag_30'] = df.groupby('State')['Total'].shift(30)

    df['rolling_mean_7'] = (
        df.groupby('State')['Total']
        .transform(lambda x: x.rolling(7).mean())
    )

    df['rolling_std_7'] = (
        df.groupby('State')['Total']
        .transform(lambda x: x.rolling(7).std())
    )

    df['day_of_week'] = df['Date'].dt.dayofweek

    df['month'] = df['Date'].dt.month

    india_holidays = holidays.India()

    df['is_holiday'] = df['Date'].isin(india_holidays)

    df = df.dropna()

    return df