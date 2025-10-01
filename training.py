#import required libraries 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from statistics import mean, stdev
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path


ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data")
data_path = DATA_DIR / "training_data.csv"

# Load the csv file
data = pd.read_csv(data_path)

# Display first 5 rows
print(data.head())

#------------------------------------------
# Data cleaning 
#------------------------------------------

# Fill missing broker_commission values
mean_commission = data.loc[data['target'] == 1, 'broker_commission'].median()
data.loc[data['target'] == 0, 'broker_commission'] = data.loc[data['target'] == 0, 'broker_commission'].fillna(0)
data.loc[data['target'] == 1, 'broker_commission'] = data.loc[data['target'] == 1, 'broker_commission'].fillna(mean_commission)

# Preprocess LOB column (split and explode)
data['LOB'] = data['LOB'].str.split(',')
data['LOB'] = data['LOB'].apply(lambda x: [i.strip() for i in x])
data = data.explode('LOB')

# Replace negative values
data['days_before_effective_date'] = data['days_before_effective_date'].abs()

#------------------------------------------
# Data preprocessing 
#------------------------------------------

# Define categorical and numerical columns
OHE_columns = ['LOB', 'New_vs_Renewal']
numerical_cols = ["broker_id", 'days_before_effective_date', 'premium_size', 'broker_commission',
                  'underwriter_turnaround_time', 'Historical_Submission_Volume']
passthrough = ["hit_ratio_broker", "Historical_Bind_Ratio"]

# Identity transformer 
identity_transformer = FunctionTransformer()  

# Define preprocessor
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown='ignore'), OHE_columns),
    ("numerical", StandardScaler(), numerical_cols),
    ("passthrough", identity_transformer, passthrough)
])

# Split features and target
X = data.drop('target', axis=1)
y = data['target']

# Fit preprocesser
preprocessor.fit(X)

cat_feature_names = preprocessor.named_transformers_['categorical'].get_feature_names_out(OHE_columns)
all_feature_names = list(cat_feature_names) + numerical_cols + passthrough

# Fit and transform input data
processed_X = preprocessor.fit_transform(X)

# Convert to DataFrame
processed_df = pd.DataFrame(processed_X, columns=all_feature_names)

# get 5 records from preprocessed data 
print(processed_df.head())

#------------------------------------------------
# Train test split and balancing training data
#------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(processed_df, y, test_size=0.3, random_state=42)
smote = SMOTE()
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)


# ------------------------------------------------------------------------------
# StratifiedKFold Cross-validation for RandomForest and XGBoost models
# ------------------------------------------------------------------------------

def stratified_cv(model, x, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc_scores = []

    for fold, (train_index, test_index) in enumerate(skf.split(x, y), 1):
        x_train_fold, x_test_fold = x.iloc[train_index], x.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        model.fit(x_train_fold, y_train_fold)
        score = model.score(x_test_fold, y_test_fold)
        acc_scores.append(score)
        print(f"Fold {fold} Accuracy: {score:.4f}")

    print("List of accuracies:", acc_scores)
    print("Max Accuracy:", max(acc_scores) * 100, "%")
    print("Min Accuracy:", min(acc_scores) * 100, "%")
    print("Mean Accuracy:", mean(acc_scores) * 100, "%")
    print("Standard Deviation:", stdev(acc_scores))

    return acc_scores

# Random Forest CV score
print("\nRandom Forest CV Results:\n")
rf_scores = stratified_cv(RandomForestClassifier(), x_train_smote, y_train_smote)

# XGBoost CV score
print("\nXGBoost CV Results:\n")
xgb_scores = stratified_cv(XGBClassifier(), x_train_smote, y_train_smote)


#----------------------------------
# Training model
#----------------------------------

# Train RandomForest model 
rf = RandomForestClassifier()
randomforest = rf.fit(x_train_smote, y_train_smote)

# Prediction from random forest model
y_pred1 = randomforest.predict(x_test)

# Random forest model evaluation
print("Confusion matrix of random forest model:\n", confusion_matrix(y_test, y_pred1))
print("Classification report of random forest model:\n", classification_report(y_test, y_pred1))
print("ROC-AUC Score:\n", roc_auc_score(y_test, y_pred1))
rf_roc_auc = roc_auc_score(y_test, y_pred1)

# Train XGBoost model
XGB = XGBClassifier()
xgb = XGB.fit(x_train_smote, y_train_smote)

# Prediction from XGBoost model 
y_pred2 = xgb.predict(x_test)

# XGBoost model evaluation
print("Confusion matrix for XGBoost model:\n", confusion_matrix(y_test, y_pred2))
print("Classification report for XGBoost model:\n", classification_report(y_test, y_pred2))
print("ROC-AUC Score:\n", roc_auc_score(y_test, y_pred2))
xgb_roc_auc = roc_auc_score(y_test, y_pred2)



# Pipelines with SMOTE
rf_pipeline = Pipeline([("preprocess", preprocessor), ("model", rf)])
xgb_pipeline = Pipeline([("preprocess", preprocessor), ("model", xgb)])

# ---------------------------
# Select best and save
# ---------------------------

best_model = rf_pipeline if rf_roc_auc >= xgb_roc_auc else xgb_pipeline
best_name = "RandomForest" if rf_roc_auc >= xgb_roc_auc else "XGBoost"

artifact_path = ARTIFACT_DIR / "model_pipeline.joblib"
joblib.dump(best_model, artifact_path)
print(f"Saved best pipeline ({best_name}) → {artifact_path}")

# ---------------------------
# Feature importance
# ---------------------------

if best_name == "RandomForest":
    features = all_feature_names
    importances = best_model.named_steps["model"].feature_importances_
    if len(features) != len(importances):
        print(f"Length mismatch: {len(features)} features vs {len(importances)} importances")
        features = [f"feature_{i}" for i in range(len(importances))]  # fallback
    importances_norm = importances / importances.sum()
    order = np.argsort(importances_norm)[::-1]
    plt.figure(figsize=(16, 8))
    plt.barh(range(len(importances_norm)), importances_norm[order], align='center')
    plt.yticks(range(len(importances_norm)), [features[i] for i in order])
    plt.xlabel("Normalized Importance Score")
    plt.title("Normalized Feature Importance (Random Forest)")
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "rf_feature_importance.png")
    plt.close()

else:
    features = all_feature_names
    gain = best_model.named_steps["model"].get_booster().get_score(importance_type="total_gain")
    importances = np.array([gain.get(f, 0.0) for f in features])
    importances_norm = importances / importances.max() if importances.max() > 0 else importances
    order = np.argsort(importances_norm)[::-1]
    plt.figure(figsize=(16, 8))
    plt.barh(range(len(importances_norm)), importances_norm[order], align='center')
    plt.yticks(range(len(importances_norm)), [features[i] for i in order])
    plt.xlabel("Normalized Importance Score")
    plt.title("Normalized Feature Importance (XGBoost)")
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "xgb_feature_importance.png")
    plt.close()
