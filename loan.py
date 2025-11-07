from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

app = Flask(__name__)

# Load or train model
def load_or_train_model():
    if os.path.exists("loan_prediction_xgboost.pkl") and os.path.exists("label_encoders.pkl"):
        return joblib.load("loan_prediction_xgboost.pkl"), joblib.load("label_encoders.pkl")

    print("🔄 Training model...")
    data = pd.read_csv("loan_data.csv")
    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
    data['Married'].fillna(data['Married'].mode()[0], inplace=True)
    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
    data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
    data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
    if 'Loan_ID' in data.columns:
        data.drop('Loan_ID', axis=1, inplace=True)

    cols = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status','Dependents']
    encoders = {}
    for col in cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "loan_prediction_xgboost.pkl")
    joblib.dump(encoders, "label_encoders.pkl")
    print("✅ Model trained and saved.")
    return model, encoders

model, encoders = load_or_train_model()

# HTML Template
template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Loan Prediction App</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      background: linear-gradient(135deg, #74ABE2, #5563DE);
      color: #fff;
    }
    .card {
      border-radius: 20px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .btn-custom {
      background: #ffcc00;
      border: none;
      color: #333;
      font-weight: bold;
    }
    .btn-custom:hover {
      background: #ffdb4d;
    }
    h2 { color: #fff; font-weight: 700; }
  </style>
</head>
<body>
<div class="container py-5">
  <h2 class="mb-4 text-center">Loan Approval Prediction</h2>
  <form method="POST" class="card p-4 bg-light text-dark shadow-lg">
    <div class="row g-3">
      <div class="col-md-3"><label>Gender</label><select name="Gender" class="form-select"><option>Male</option><option>Female</option></select></div>
      <div class="col-md-3"><label>Married</label><select name="Married" class="form-select"><option>Yes</option><option>No</option></select></div>
      <div class="col-md-3"><label>Dependents</label><select name="Dependents" class="form-select"><option>0</option><option>1</option><option>2</option><option>3+</option></select></div>
      <div class="col-md-3"><label>Education</label><select name="Education" class="form-select"><option>Graduate</option><option>Not Graduate</option></select></div>
      <div class="col-md-3"><label>Self Employed</label><select name="Self_Employed" class="form-select"><option>No</option><option>Yes</option></select></div>
      <div class="col-md-3"><label>Property Area</label><select name="Property_Area" class="form-select"><option>Urban</option><option>Semiurban</option><option>Rural</option></select></div>
      <div class="col-md-3"><label>Applicant Income</label><input type="number" name="ApplicantIncome" value="5000" class="form-control" required></div>
      <div class="col-md-3"><label>Coapplicant Income</label><input type="number" name="CoapplicantIncome" value="2000" class="form-control" required></div>
      <div class="col-md-3"><label>Loan Amount</label><input type="number" name="LoanAmount" value="50000" class="form-control" required></div>
      <div class="col-md-3"><label>Loan Term (Days)</label><input type="number" name="Loan_Amount_Term" value="360" class="form-control" required></div>
      <div class="col-md-3"><label>Credit History (1 = Good)</label><select name="Credit_History" class="form-select"><option value="1">1</option><option value="0">0</option></select></div>
    </div>
    <div class="text-center mt-4">
      <button class="btn btn-custom px-4">Predict</button>
    </div>
  </form>

  {% if result is defined %}
  <div class="alert mt-4 {% if result=='Approved' %}alert-success{% else %}alert-danger{% endif %} text-center">
    <h4 class="fw-bold">Prediction: {{ result }}</h4>
    {% if prob %}<p>Approval Probability: {{ prob }}%</p>{% endif %}
    {% if reason %}<p><b>Reason:</b> {{ reason }}</p>{% endif %}
  </div>
  {% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result, prob, reason = None, None, None
    if request.method == "POST":
        form = request.form
        df = pd.DataFrame([{
            'Gender': form['Gender'], 'Married': form['Married'], 'Dependents': form['Dependents'],
            'Education': form['Education'], 'Self_Employed': form['Self_Employed'],
            'ApplicantIncome': float(form['ApplicantIncome']),
            'CoapplicantIncome': float(form['CoapplicantIncome']),
            'LoanAmount': float(form['LoanAmount']),
            'Loan_Amount_Term': float(form['Loan_Amount_Term']),
            'Credit_History': float(form['Credit_History']),
            'Property_Area': form['Property_Area']
        }])

        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])

        pred = model.predict(df)[0]
        prob = round(model.predict_proba(df)[0][1]*100, 2)
        result = "Approved" if pred == 1 else "Rejected"

        # Reasoning for rejection
        if result == "Rejected":
            income = df['ApplicantIncome'].values[0]
            loan_amt = df['LoanAmount'].values[0]
            credit = df['Credit_History'].values[0]
            self_emp = df['Self_Employed'].values[0]
            ratio = loan_amt / (income + 1)

            if credit == 0:
                reason = "Low or no credit history detected."
            elif ratio > 0.05:
                reason = "Loan amount is too high compared to your income."
            elif income < 4000:
                reason = "Applicant income too low for requested loan."
            elif df['Loan_Amount_Term'].values[0] > 360:
                reason = "Very long loan term requested."
            elif self_emp == 1 and credit == 0:
                reason = "Self-employed without sufficient credit history."
            else:
                reason = "Overall application risk too high based on data."

    return render_template_string(template, result=result, prob=prob, reason=reason)

# Entry point for both local + Vercel
if __name__ == "__main__":
    app.run(debug=True)
else:
    app = app
