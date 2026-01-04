import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DataFile = os.path.join(BASE_DIR, "DiabetesData.csv")

data = pd.read_csv(DataFile)

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train, y_train)

# Save model & scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained & saved")
