from src.preprocess import load_and_clean_data
from src.feature_engineering import create_features
from src.train_models import train_models
from src.evaluate_models import save_results

file_path = 'data/sales_data.xlsx'

df = load_and_clean_data(file_path)

df = create_features(df)

results = train_models(df)

save_results(results)
