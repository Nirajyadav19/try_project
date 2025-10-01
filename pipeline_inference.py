import pandas as pd
import joblib
from pathlib import Path
 
ARTIFACT_DIR = Path("artifacts") 
pipeline = joblib.load(ARTIFACT_DIR / "model_pipeline.joblib")
DATA_DIR = Path("data")
data_path1 = DATA_DIR / "inference_data.csv"
data_path2 = DATA_DIR / "inference_testing_data.csv"
# Load the csv file
data = pd.read_csv(data_path1)
inference_data=pd.read_csv(data_path2)
print(data)
inference_data['LOB'] = inference_data['LOB'].str.split(',')
inference_data['LOB'] = inference_data['LOB'].apply(lambda x: [i.strip() for i in x])
inference_data = inference_data.explode('LOB')
inference_data['broker_commission'] = inference_data['broker_commission'].fillna(0)
inference_data['days_before_effective_date'] = inference_data['days_before_effective_date'].abs()

y_pred = pipeline.predict(inference_data)

# Add predictions to inference_data
inference_data['y_pred'] = y_pred

print(inference_data)
