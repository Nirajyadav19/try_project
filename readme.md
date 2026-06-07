Project Overview

This project builds and evaluates a machine learning model to predict a binary target variable using structured insurance submission data. Two models — RandomForestClassifier and XGBoostClassifier — are trained, validated
and compared using Stratified K-Fold cross-validation. The best-performing model is saved and used for inference on unseen validation data.

1. Training the Model:
Run the training.py script :
      To execute the training.py on the training dataset, run the following command in your terminal:  
	python training.py

Artifacts generated:
    model_pipeline.joblib → saved pipeline including preprocessing and best model
    xgb_feature_importance.png 


2. Testing the Model:
Run the testing.py script :
     To execute the testing.py on the validation dataset, run the following command in your terminal:  
	python testing.py

Evaluation metrics generated:
    Accuracy
    Classification Report
    Confusion Matrix
