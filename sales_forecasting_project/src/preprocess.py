import pandas as pd

def load_and_clean_data(file_path):

    df = pd.read_excel(file_path)

    df.columns = [col.strip() for col in df.columns]

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values(by=['State', 'Date'])

    all_states = []

    for state in df['State'].unique():

        temp = df[df['State'] == state].copy()

        temp = temp.set_index('Date')

        temp = temp.asfreq('D')

        temp['State'] = state

        temp['Total'] = temp['Total'].ffill()

        all_states.append(temp)

    df = pd.concat(all_states).reset_index()

    return df