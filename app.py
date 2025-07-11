import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Flask application setup
application = Flask(__name__)
app = application

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction page
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Collect form data
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),  # ✅ FIXED
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                writing_score=float(request.form.get('writing_score')),
                reading_score=float(request.form.get('reading_score'))
            )

            # Convert to DataFrame
            pred_df = data.get_data_as_dataframe()
            print("Input DataFrame:\n", pred_df)

            # Run prediction
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            # Return result
            return render_template('home.html', results=results[0])

        except Exception as e:
            print("❌ Prediction Error:", e)
            return render_template('home.html', results=f"❌ Prediction failed: {e}")

# Main runner
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # ✅ Debug mode for development
