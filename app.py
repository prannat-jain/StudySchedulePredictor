from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("student_performance_model.pkl")

# Load the column order
columns_order = joblib.load("columns_order.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input data from the form
        data_dict = {
            "StudyTimeWeekly": float(request.form["study_time"]),
            "Absences": int(request.form["absences"]),
            "Tutoring": int(request.form["tutoring"]),
            "ParentalSupport": int(request.form["parental_support"]),
            "Extracurricular": int(request.form["extracurricular"]),
            "Sports": int(request.form["sports"]),
            "Music": int(request.form["music"]),
            "Volunteering": int(request.form["volunteering"]),
            "Age": int(request.form["age"]),
            "ParentalEducation": int(request.form["parental_education"]),
            "Gender_1": int(request.form["gender"]),
            "Ethnicity_1": int(request.form["ethnicity"]) == 1,
            "Ethnicity_2": int(request.form["ethnicity"]) == 2,
            "Ethnicity_3": int(request.form["ethnicity"]) == 3
        }

        # Create a DataFrame with the input data
        input_df = pd.DataFrame([data_dict])

        # Reorder columns to match training data
        input_df = input_df.reindex(columns=columns_order, fill_value=0)

        # Predict using the model
        prediction = model.predict(input_df)

        return render_template("index.html", prediction=int(prediction[0]))

    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
