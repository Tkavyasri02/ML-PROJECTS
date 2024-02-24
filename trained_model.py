import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# Load the dataset
data = pd.read_excel(r'C:\Users\Dell\OneDrive\Documents\c c++\python_for_web\Optimizing-Agricultural-Production-XGBOOST-main\data.xlsx')

# Train-test split
y = data['label']
x = data.drop(['label'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Logistic Regression model with a different solver
logistic_model = LogisticRegression(solver='saga',max_iter=1000)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
logistic_model.fit(x_train_scaled, y_train)

# Save logistic regression model to a pickle file
joblib.dump(logistic_model, 'logistic_model.pkl')

# XGBoost model
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

x_train_encoded = pd.get_dummies(x_train)
x_test_encoded = pd.get_dummies(x_test)

dtrain = xgb.DMatrix(x_train_encoded, label=y_train_encoded)
dtest = xgb.DMatrix(x_test_encoded, label=y_test_encoded)

params = {
    'objective': 'multi:softmax',
    'num_class': len(data['label'].unique()),
    'eval_metric': 'mlogloss',
}

num_round = 100
xg_model = xgb.train(params, dtrain, num_round)

# Save XGBoost model to a pickle file
joblib.dump(xg_model, 'xgboost_model.pkl')

