from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import tensorflow_decision_forests as tfdf
from tensorflow import keras

app = Flask(__name__)
options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
model = tf.keras.models.load_model('save_model/1', options=options)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.values()
    float_features = [float(x) for x in data]
    final_features = np.array(float_features).reshape(1, -1)
    predictions = model.predict(final_features)

    predicted_class_index = np.argmax(predictions)
    class_labels = ["NPK 10-26-26", "NPK 14-35-14", "NPK 17-17-17", "NPK 20-20", "NPK 28-28", "NPK DAP", "NPK Urea"]
    predicted_class = class_labels[predicted_class_index]

    return jsonify({"predictions": predicted_class})

@app.route("/predict-web", methods=["POST"])
def predictWeb():
    float_features = [float(x) for x in request.form.values()]
    final_features = np.array(float_features).reshape(1, -1)
    predictions = model.predict(final_features)

    predicted_class_index = np.argmax(predictions)
    class_labels = ["NPK 10-26-26", "NPK 14-35-14", "NPK 17-17-17", "NPK 20-20", "NPK 28-28", "NPK DAP", "NPK Urea"]
    predicted_class = class_labels[predicted_class_index]

    return render_template("index.html", prediction_text="{}".format(predicted_class))

if __name__ == "__main__":
    app.run(debug=True)
