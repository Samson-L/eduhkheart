from flask import Flask, render_template, request
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# ---------- Data Preparation & Model Training (runs at startup) ----------

df = pd.read_csv('heart_attack_china-4.csv')

features = [
    "Age", "Gender", "Smoking_Status", "Hypertension", "Diabetes", "Obesity",
    "Cholesterol_Level", "Air_Pollution_Exposure", "Physical_Activity", "Diet_Score",
    "Stress_Level", "Alcohol_Consumption", "Family_History_CVD", "Healthcare_Access",
    "Rural_or_Urban", "Region", "Hospital_Availability", "TCM_Use", "Employment_Status",
    "Education_Level", "Income_Level", "Blood_Pressure", "Chronic_Kidney_Disease",
    "Previous_Heart_Attack", "CVD_Risk_Score",
]
categorical = [
    "Gender", "Smoking_Status", "Hypertension", "Diabetes", "Obesity", "Cholesterol_Level",
    "Air_Pollution_Exposure", "Physical_Activity", "Diet_Score", "Stress_Level", "Alcohol_Consumption",
    "Family_History_CVD", "Healthcare_Access", "Rural_or_Urban", "Region", "Hospital_Availability",
    "TCM_Use", "Employment_Status", "Education_Level", "Income_Level", "Chronic_Kidney_Disease",
    "Previous_Heart_Attack",
]

X = df[features].copy()
y = df['Heart_Attack']

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
X[categorical] = imputer.fit_transform(X[categorical])

# Store LabelEncoders for all categorical variables
encoders = {}
for col in categorical:
    enc = LabelEncoder()
    X[col] = enc.fit_transform(X[col])
    encoders[col] = enc

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM model
svm = SVC(probability=True)
svm.fit(X_scaled, y)

# ---------- Flask Routes ----------

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None

    # Show the form and handle input
    if request.method == 'POST':
        # Collect form data
        user_input = {feature: request.form.get(feature) for feature in features}
        # Convert numeric fields
        for field in ["Age", "Blood_Pressure", "CVD_Risk_Score"]:
            user_input[field] = float(user_input[field])

        user_df = pd.DataFrame([user_input])

        # Encode categorical variables using fitted encoders
        for col in categorical:
            user_df[col] = encoders[col].transform(user_df[col])

        # Scale
        user_scaled = scaler.transform(user_df)
        pred = svm.predict(user_scaled)[0]
        prob = svm.predict_proba(user_scaled)[0][1]

        prediction = 'May Have a Heart Attack' if pred == 1 else 'Unlikely to Have a Heart Attack'
        probability = f"{prob*100:.2f}%"

    # Render form
    # For select dropdowns, get choices from training data
    choices = {col: sorted(df[col].dropna().unique()) for col in categorical}

    return render_template(
        'form.html',
        features=features,
        categorical=categorical,
        choices=choices,
        prediction=prediction,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)