#app.py

from flask import Flask, render_template, request
import numpy as np
import joblib
import xgboost as xgb
import os

app = Flask(__name__)

# Check if the pickled model files exist
if not os.path.exists('logistic_model.pkl') or not os.path.exists('xgboost_model.json'):
    print("Model files not found.")
    exit()

# Load the logistic regression model
logistic_model = joblib.load('logistic_model.pkl')

# Load the XGBoost model
xg_model = xgb.Booster()
xg_model.load_model('xgboost_model.json')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the features from the form
    features = [float(x) for x in request.form.values()]
    input_data = np.array(features).reshape(1, -1)
    
    # Make prediction using logistic regression model
    logistic_prediction = logistic_model.predict(input_data)
    
    # Make prediction using XGBoost model
    xg_prediction = xg_model.predict(xgb.DMatrix(input_data))
    
    return render_template('result.html', logistic_prediction=logistic_prediction, xg_prediction=xg_prediction)

if __name__ == '__main__':
    app.run(debug=True)




