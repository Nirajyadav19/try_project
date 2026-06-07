from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# Load model pipeline once
ARTIFACT_DIR = Path("artifacts")
pipeline = joblib.load(ARTIFACT_DIR / "model_pipeline.joblib")
lob_categories = joblib.load(ARTIFACT_DIR /'lob_categories.joblib')

def preprocess_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Split and strip the 'LOB' strings into lists
    df['LOB'] = df['LOB'].apply(lambda x: [lob.strip() for lob in x.split(',')])
    
    # Create binary columns in fixed order
    for lob in lob_categories:
        df[lob] = df['LOB'].apply(lambda x: int(lob in x))
    df = df.drop(columns=['LOB'],axis=1)

    # Step 3: Preprocessing
    df['broker_commission'] = df['broker_commission'].fillna(0)
    df['days_before_effective_date'] = df['days_before_effective_date'].abs()

    # Step 4: Predict for each exploded row
    y_pred = pipeline.predict(df)
    df['y_pred'] = y_pred
    df['y_pred'] = df['y_pred'].map({0: "Not Bound", 1: "Bound"})

    # Reconstruct LOB from binary columns
    def reconstruct_lob(row):
        present_lobs = [lob for lob in lob_categories if row[lob] == 1]
        return ", ".join(present_lobs)
    
    df['LOB'] = df.apply(reconstruct_lob, axis=1)

    df = df.drop(columns=lob_categories)

    return df[['LOB', 'New_vs_Renewal', 'broker_commission', 'broker_id', 'premium_size', 'y_pred']]

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

        final_result=result_df[["LOB","New_vs_Renewal","broker_commission","broker_id","premium_size","y_pred"]]
        
        result_json = final_result.to_dict(orient='records')
 
        # Convert to JSON
        return jsonify(result_json), 200
  
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
