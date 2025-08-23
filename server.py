# server.py
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('customer_churn_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Read CSV file into DataFrame
    df = pd.read_csv(file)
    df.fillna(0, inplace=True)
    
    # Preprocess input data (same as training)
    X_input = df.drop(['customer_id', 'phone_no'], axis=1)  # Drop non-feature columns
    X_input = pd.get_dummies(X_input)  # Convert categorical variables
    
    # Align columns with the training set (in case of missing categories)
    X_input = X_input.reindex(columns=model.feature_importances_.index.tolist(), fill_value=0)

    # Make predictions
    predictions = model.predict(X_input)
    
    # Prepare response data for visualization (counts of churned vs not churned)
    churn_count = np.sum(predictions)
    not_churn_count = len(predictions) - churn_count
    
    return jsonify({
        'predictions': predictions.tolist(),
        'churn_count': churn_count,
        'not_churn_count': not_churn_count,
        'total_count': len(predictions)
    })

if __name__ == '__main__':
    app.run(debug=True)