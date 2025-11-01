import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler  # type: ignore

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    else:
        data = CustomData(
            gender=request.form["gender"],
            race_ethnicity=request.form["ethnicity"],
            parental_level_of_education=request.form[
                "parental_level_of_education"
            ],
            lunch=request.form["lunch"],
            test_preparation_course=request.form["test_preparation_course"],
            reading_score=float(request.form["reading_score"]),
            writing_score=float(request.form["writing_score"]),
        )
        prediction_df = data.get_data_as_data_frame()
        print(prediction_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(prediction_df)
        return render_template("home.html", results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
