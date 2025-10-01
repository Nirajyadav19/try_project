from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# Load model pipeline once
ARTIFACT_DIR = Path("artifacts")
pipeline = joblib.load(ARTIFACT_DIR / "model_pipeline.joblib")

def preprocess_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    # Preprocessing same as your original code
    df['LOB'] = df['LOB'].str.split(',')
    df['LOB'] = df['LOB'].apply(lambda x: [i.strip() for i in x])
    df = df.explode('LOB')
    df['broker_commission'] = df['broker_commission'].fillna(0)
    df['days_before_effective_date'] = df['days_before_effective_date'].abs()

    #prediction & Add prediction column 
    y_pred=pipeline.predict(df)
    df["y_pred"] = y_pred
    df['y_pred'] = df['y_pred'].map({0: "Not Bound", 1: "Bound"})

    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request body
        input_json = request.get_json()

        if not input_json:
            return jsonify({"error": "Empty JSON input"}), 400
        
        # Convert JSON to DataFrame
        df = pd.DataFrame(input_json)

        # Check required columns exist
        required_cols = ["LOB","broker_id","hit_ratio_broker","days_before_effective_date","premium_size","broker_commission","underwriter_turnaround_time","New_vs_Renewal","Historical_Submission_Volume","Historical_Bind_Ratio"]
        for col in required_cols:
            if col not in df.columns:
                return jsonify({"error": f"Missing required column: {col}"}), 400

        # Run preprocessing + prediction
        result_df = preprocess_and_predict(df)
        print(f"result df {result_df}")

        final_result=result_df[["LOB","New_vs_Renewal","broker_commission","broker_id","premium_size","y_pred"]]
        
        result_json = final_result.to_dict(orient='records')
 
        # Convert to JSON
        return jsonify(result_json)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
