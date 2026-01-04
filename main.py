# COPYRIGHT (C) © HTTPS://WWW.COMPUTES.COM 2024 . ALL RIGHTS RESERVED.......
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
CORS(app)

@app.route("/")
def Home() :
    return {"message": "AIM Backend is Live!!!"}

@app.route("/API", methods=["GET", "POST"])
def API() :
    if request.method == "POST" :
        PN = float(request.json.get("Pregnancies"))
        GL = float(request.json.get("Glucose"))
        BP = float(request.json.get("BloodPressure"))
        ST = float(request.json.get("SkinThickness"))
        INSU = float(request.json.get("Insulin"))
        BMI = float(request.json.get("BMI"))
        DPF = float(request.json.get("DiabetesPedigreeFunction")) 
        AGE = float(request.json.get("Age"))

        jsonify(PN, GL, BP, ST, INSU, BMI, DPF, AGE)

        # it has a 'target' column where 1 indicates diabetic and 0 indicates non-diabetic
        DataFile = "/static/DibetesData.CSV"
        Data = pd.read_csv(DataFile)

        # Split the Data into features and target
        X = Data.drop('Outcome', axis=1)
        y = Data['Outcome']

        # Split the Data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the Data (important for models like Logistic Regression)
        Scaler = StandardScaler()
        X_train = Scaler.fit_transform(X_train)
        X_test = Scaler.transform(X_test)

        # Create and train the Logistic Regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

        # Classification report
        print(classification_report(y_test, y_pred))

        # If you want to make a Prediction on a new Data point:
        DataPoints = [[PN, GL, BP, ST, INSU, BMI, DPF, AGE]]

        NewDataPoints = Scaler.transform(DataPoints)

        Prediction = model.predict(NewDataPoints)
        
        Data = {"Data": Prediction.tolist()}

        try:
            return jsonify({"Status": "Success", "Result": Data})

        except Exception as e :
            print(e)

    else :
        return render_template("API-HANDLE.HTML")

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=8000, debug=True)