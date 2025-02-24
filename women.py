import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv("Dataset - Updated.csv")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist() 
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median()) 

categorical_cols = df.select_dtypes(include=[object]).columns.tolist()  
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])  # Fill categorical NaN with mode

# Define feature columns and target variable
FEATURES = ['Age', 'Systolic BP', 'Diastolic', 'BS', 'Body Temp', 'BMI', 'Previous Complications',
            'Preexisting Diabetes', 'Gestational Diabetes', 'Mental Health', 'Heart Rate']
TARGET = 'Risk Level'

df[TARGET] = df[TARGET].map({'High': 1, 'Low': 0})

X = df[FEATURES]
y = df[TARGET]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "women_risk_model.pkl")

print("Model training complete. Model saved as women_risk_model.pkl")
