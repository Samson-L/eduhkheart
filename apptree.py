import pandas as pd
from flask import Flask, render_template_string, request
from sklearn.tree import DecisionTreeRegressor
from flask_ngrok import run_with_ngrok

# ===== MODEL TRAINING =====
train_df = pd.read_csv('europeanheart-train.csv')

FEATURES = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal'
]

X_train = train_df[FEATURES]
y_train = train_df['target']

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# ===== FLASK APP =====

app = Flask(__name__)

run_with_ngrok(app)
# Metadata for user-friendly form
FEATURE_META = [
    {
        "name": "age",
        "label": "Age",
        "type": "number",
        "placeholder": "e.g., 52",
        "desc": "Your age in years (numeric)."
    },
    {
        "name": "sex",
        "label": "Sex",
        "type": "select",
        "options": [
            {"value": 0, "text": "Female"},
            {"value": 1, "text": "Male"},
        ],
        "desc": "0 = Female, 1 = Male"
    },
    {
        "name": "cp",
        "label": "Chest Pain Type",
        "type": "select",
        "options": [
            {"value": 0, "text": "Typical Angina"},
            {"value": 1, "text": "Atypical Angina"},
            {"value": 2, "text": "Non-anginal Pain"},
            {"value": 3, "text": "Asymptomatic"},
        ],
        "desc": "Types: 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic"
    },
    {
        "name": "trestbps",
        "label": "Resting Blood Pressure",
        "type": "number",
        "placeholder": "e.g., 125",
        "desc": "Resting blood pressure (in mm Hg on admission)."
    },
    {
        "name": "chol",
        "label": "Serum Cholesterol",
        "type": "number",
        "placeholder": "e.g., 212",
        "desc": "Serum cholesterol in mg/dL (numeric)."
    },
    {
        "name": "fbs",
        "label": "Fasting Blood Sugar",
        "type": "select",
        "options": [
            {"value": 0, "text": "<= 120 mg/dL"},
            {"value": 1, "text": "> 120 mg/dL"},
        ],
        "desc": "0 = ≤120 mg/dL, 1 = >120 mg/dL"
    },
    {
        "name": "restecg",
        "label": "Resting Electrocardiographic Results",
        "type": "select",
        "options": [
            {"value": 0, "text": "Normal"},
            {"value": 1, "text": "Abnormality"},
            {"value": 2, "text": "Hypertrophy"},
        ],
        "desc": "0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy"
    },
    {
        "name": "thalach",
        "label": "Maximum Heart Rate Achieved",
        "type": "number",
        "placeholder": "e.g., 168",
        "desc": "Maximum heart rate achieved (numeric)."
    },
    {
        "name": "exang",
        "label": "Exercise-Induced Angina",
        "type": "select",
        "options": [
            {"value": 0, "text": "No"},
            {"value": 1, "text": "Yes"},
        ],
        "desc": "0 = No, 1 = Yes"
    },
    {
        "name": "oldpeak",
        "label": "ST Depression (Oldpeak)",
        "type": "number",
        "step": "any",
        "placeholder": "e.g., 1.0",
        "desc": "ST depression induced by exercise relative to rest (numeric, e.g., 1.0)."
    },
    {
        "name": "slope",
        "label": "Slope of Peak Exercise ST Segment",
        "type": "select",
        "options": [
            {"value": 0, "text": "Upsloping"},
            {"value": 1, "text": "Flat"},
            {"value": 2, "text": "Downsloping"},
        ],
        "desc": "0 = Upsloping, 1 = Flat, 2 = Downsloping"
    },
    {
        "name": "ca",
        "label": "Number of Major Vessels (0-3) Colored by Fluoroscopy",
        "type": "number",
        "min": 0,
        "max": 3,
        "placeholder": "e.g., 0",
        "desc": "Integer from 0 to 3."
    },
    {
        "name": "thal",
        "label": "Thalassemia",
        "type": "select",
        "options": [
            {"value": 1, "text": "Normal"},
            {"value": 2, "text": "Fixed Defect"},
            {"value": 3, "text": "Reversible Defect"},
        ],
        "desc": "1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect"
    },
]

form_html = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>European Heart Risk Predictor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
      body { font-family: sans-serif; margin: 2em; }
      form { max-width: 500px; margin: auto; }
      label { font-weight: bold; }
      .desc { font-size: 0.95em; color: #555; margin-bottom: 0.5em;}
      input, select { width: 100%; margin-bottom: 1em; padding: 0.5em; }
      .result { margin: 2em 0; padding: 1em; background: #e6ffe6; border-radius: 5px;}
      @media (max-width: 600px) {
        form { width: 100%; }
      }
  </style>
</head>
<body>
  <h1>European Heart Risk Predictor</h1>
  <form method="post" autocomplete="off">
    {% for feat in FEATURE_META %}
      <label for="{{ feat.name }}">{{ feat.label }}</label>
      {% if feat.type == "select" %}
        <select name="{{ feat.name }}" id="{{ feat.name }}" required>
          <option value="" disabled selected>Select...</option>
          {% for opt in feat.options %}
            <option value="{{ opt.value }}">{{ opt.text }}</option>
          {% endfor %}
        </select>
      {% else %}
        <input 
          type="number" 
          name="{{ feat.name }}" 
          id="{{ feat.name }}" 
          {% if feat.placeholder %}placeholder="{{ feat.placeholder }}"{% endif %}
          {% if feat.step %}step="{{ feat.step }}"{% endif %}
          {% if feat.min is defined %}min="{{ feat.min }}"{% endif %}
          {% if feat.max is defined %}max="{{ feat.max }}"{% endif %}
          required
        >
      {% endif %}
      <div class="desc">{{ feat.desc }}</div>
    {% endfor %}
    <button type="submit">Predict Risk</button>
  </form>
  {% if prediction is not none %}
    <div class="result">
      <strong>Predicted risk score:</strong> {{ prediction }}
    </div>
  {% endif %}
  <footer style="margin-top:3em;font-size:85%;color:#888;">
      &copy; Your Heart Risk App
  </footer>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            values = []
            for feat in FEATURE_META:
                val = request.form[feat["name"]]
                # Cast to float for model input
                values.append(float(val))
            df_input = pd.DataFrame([values], columns=FEATURES)
            pred = model.predict(df_input)[0]
            prediction = f"{pred:.2f}"
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template_string(
        form_html,
        FEATURE_META=FEATURE_META,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)
