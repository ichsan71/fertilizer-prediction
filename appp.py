from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('classifier.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    
    return render_template("index.html", request.form.values())

    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    predictions = model.predict(final_features)

    if predictions == 0:
        predictions = "NPK 10-26-26"
    elif predictions == 1:
        predictions = "NPK 14-35-14"
    elif predictions == 2:
        predictions = "NPK 17-17-17"
    elif predictions == 3:
        predictions = "NPK 20-20"
    elif predictions == 4:
        predictions = "NPK 28-28"
    elif predictions == 5:
        predictions = "NPK DAP"
    else:
        predictions = "NPK Urea"

    return render_template("index.html", prediction_text="{}".format(predictions))

if __name__ == "__main__":
    app.run(debug=True)
