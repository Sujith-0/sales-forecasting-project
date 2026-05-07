import pandas as pd

def save_results(results):

    df = pd.DataFrame(
        list(results.items()),
        columns=['Model', 'MAE']
    )

    df.to_csv('outputs/model_results.csv', index=False)

    print(df)
