from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from joblib import load
# Add to app.py
import numpy as np
print("DEPLOYMENT NUMPY VERSION:", np.__version__)

app = Flask(__name__)

# Load artifacts
model = load('knn_model.joblib')  # Replace with your actual model filename
scaler = load('heart_scaler.joblib')  # Fixed filename

# Load feature names
with open('feature_names.txt', 'r') as f:
    expected_columns = f.read().splitlines()

# Feature mapping (update with your actual categories)
numerical_cols = ['age', 'resting_blood_pressure', 'serum_cholestoral', 
                 'max_heart_rate', 'oldpeak']
categorical_cols = {
    'chest_pain_type': ['1', '2', '3', '4'],
    'resting_electrocardiographic_results': ['0', '1', '2'],
    'ST_segment': ['1', '2', '3'],
    'major_vessels': ['0', '1', '2', '3'],
    'thal': ['3', '6', '7']
}

@app.route('/')
def home():
    return render_template('index.html',
                         numerical_cols=numerical_cols,
                         categorical_cols=categorical_cols)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        input_df = pd.DataFrame([data])
        
        # Process numerical features
        input_df[numerical_cols] = input_df[numerical_cols].astype(float)
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        # Process categorical features
        for col, values in categorical_cols.items():
            for val in values:
                input_df[f"{col}_{val}"] = (input_df[col] == val).astype(int)
            input_df.drop(col, axis=1, inplace=True)
        
        # Ensure all expected columns exist
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'message': 'High risk' if prediction == 1 else 'Low risk'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
