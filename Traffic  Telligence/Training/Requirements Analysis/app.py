import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='template')

# Load model and encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    # Get input values from HTML form
    input_feature = [x for x in request.form.values()]
    # Define original feature names
    original_features = ["holiday", "temp", "rain", "snow", "weather", "year", "month", "day", "hours", "minutes", "seconds"]
    
    # Convert to DataFrame
    data = pd.DataFrame([input_feature], columns=original_features)

    # Convert numeric columns to float
    numeric_features = ["temp", "rain", "snow", "year", "month", "day", "hours", "minutes", "seconds"]
    data[numeric_features] = data[numeric_features].astype(float)

    # Define categorical columns
    categorical_features = ["holiday", "weather"]

    # Apply OneHotEncoder to categorical columns
    encoded_features = encoder.transform(data[categorical_features])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

    # Merge encoded and numeric data
    data = data.drop(columns=categorical_features).reset_index(drop=True)
    data = pd.concat([data, encoded_df], axis=1)

    # Match model input structure
    data = data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict and render result
    prediction = model.predict(data)
    result_text = "Estimated Traffic Volume: " + str(int(prediction[0]))
    return render_template("output.html", result=result_text)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
