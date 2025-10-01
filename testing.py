import pandas as pd
import joblib
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from pathlib import Path
 
ARTIFACT_DIR = Path("artifacts")
pipeline = joblib.load(ARTIFACT_DIR / "model_pipeline.joblib")
DATA_DIR = Path("data")
data_path = DATA_DIR / "validation.csv"

# Load the csv file
data = pd.read_csv(data_path)

data['LOB'] = data['LOB'].str.split(',')
data['LOB'] = data['LOB'].apply(lambda x: [i.strip() for i in x])
data = data.explode('LOB')
mean_commission = data.loc[data['target'] == 1, 'broker_commission'].median()
data.loc[data['target'] == 0, 'broker_commission'] = data.loc[data['target'] == 0, 'broker_commission'].fillna(0)
data.loc[data['target'] == 1, 'broker_commission'] = data.loc[data['target'] == 1, 'broker_commission'].fillna(mean_commission)
data['days_before_effective_date'] = data['days_before_effective_date'].abs()

x = data.drop('target', axis=1)
y = data['target']
y_pred = pipeline.predict(x)

print("accuracy score\n",accuracy_score(y,y_pred))
print(classification_report(y, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

data['predict']=y_pred
print(data.head())