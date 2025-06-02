import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('heart_attack_china-4.csv')

# Select features for prediction
features = [
    "Age",
    "Gender",
    "Smoking_Status",
    "Hypertension",
    "Diabetes",
    "Obesity",
    "Cholesterol_Level",
    "Air_Pollution_Exposure",
    "Physical_Activity",
    "Diet_Score",
    "Stress_Level",
    "Alcohol_Consumption",
    "Family_History_CVD",
    "Healthcare_Access",
    "Rural_or_Urban",
    "Region",
    "Hospital_Availability",
    "TCM_Use",
    "Employment_Status",
    "Education_Level",
    "Income_Level",
    "Blood_Pressure",
    "Chronic_Kidney_Disease",
    "Previous_Heart_Attack",
    "CVD_Risk_Score",
]

categorical = [
    "Gender",
    "Smoking_Status",
    "Hypertension",
    "Diabetes",
    "Obesity",
    "Cholesterol_Level",
    "Air_Pollution_Exposure",
    "Physical_Activity",
    "Diet_Score",
    "Stress_Level",
    "Alcohol_Consumption",
    "Family_History_CVD",
    "Healthcare_Access",
    "Rural_or_Urban",
    "Region",
    "Hospital_Availability",
    "TCM_Use",
    "Employment_Status",
    "Education_Level",
    "Income_Level",
    "Chronic_Kidney_Disease",
    "Previous_Heart_Attack",
]

X = df[features]
y = df['Heart_Attack']



# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')


# Encode categorical variables
for i in categorical:
    print(f'ran {i}')
    X[i] = LabelEncoder().fit_transform(X[i])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data for training/testing
#X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVM model
svm = SVC(probability=True)
svm.fit(X_scaled, y)

# Prepare user input for prediction received from the frontend form
"""
user_input = {
    "Age": 30, 
    "Gender": gender, 
    "Smoking_Status": smoking_status, 
    "Hypertension": hypertension, 
    "Diabetes": diabetes, 
    "Obesity": obesity, 
    "Cholesterol_Level": cholesterol_level, 
    "Air_Pollution_Exposure": air_pollution_exposure,  
    "Physical_Activity": physical_activity,  
    "Diet_Score": diet_score, 
    "Stress_Level": stress_level, 
    "Alcohol_Consumption": alcohol_consumption,  
    "Family_History_CVD": family_history_cvd,  
    "Healthcare_Access": healthcare_access,  
    "Rural_or_Urban": rural_or_urban,  
    "Region": region,  
    "Hospital_Availability": hospital_availability,  
    "TCM_Use": tcm_use,  
    "Employment_Status": employment_status,  
    "Education_Level": education_level,  
    "Income_Level": income_level,  
    "Blood_Pressure": blood_pressure,  
    "Chronic_Kidney_Disease": chronic_kidney_disease,  
    "Previous_Heart_Attack": previous_heart_attack,  
    "CVD_Risk_Score": cvd_risk_score,  
}
"""

patient_data = {
    "Age": 55,
    "Gender": "Male",
    "Smoking_Status": "Non-Smoker",
    "Hypertension": "No",
    "Diabetes": "No",
    "Obesity": "Yes",
    "Cholesterol_Level": "Normal",
    "Air_Pollution_Exposure": "High",
    "Physical_Activity": "High",
    "Diet_Score": "Moderate",
    "Stress_Level": "Low",
    "Alcohol_Consumption": "Yes",
    "Family_History_CVD": "No",
    "Healthcare_Access": "Good",
    "Rural_or_Urban": "Rural",
    "Region": "Eastern",
    "Hospital_Availability": "Low",
    "TCM_Use": "Yes",
    "Employment_Status": "Unemployed",
    "Education_Level": "Primary",
    "Income_Level": "Low",
    "Blood_Pressure": 104,
    "Chronic_Kidney_Disease": "Yes",
    "Previous_Heart_Attack": "No",
    "CVD_Risk_Score": 78,
}



# Convert to DataFrame
user_df = pd.DataFrame([patient_data])

# Encode categorical variables
for i in categorical:
    print(f"ran on {i}")
    user_df[i] = LabelEncoder().fit(df[i]).transform(user_df[i])


print('executed')
# Scale the data
user_scaled = scaler.transform(user_df)

# Predict survival
prediction = svm.predict(user_scaled)[0]

prob = svm.predict_proba(user_scaled)[0][1] 
print(svm.predict_proba(user_scaled))
# Show result in a pop-up window
result = 'May Have a heart attack' if prediction == 1 else 'Wont likely to Have a Heart Attack'
print(f"Heart Attack Prediction: {result}\nPredicted Probability of Heart attack: {prob:.2f}")


