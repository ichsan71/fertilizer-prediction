from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import tensorflow_decision_forests as tfdf
from tensorflow import keras

app = Flask(__name__)
options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
model = tf.keras.models.load_model('./save_model/1', options=options)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form

    # Prepare the input data for prediction
    input_data = {
        'crop_type': tf.constant([data['crop_type']]),
        'humidity': tf.constant([int(data['humidity'])], dtype=tf.int64),
        'moisture': tf.constant([int(data['moisture'])], dtype=tf.int64),
        'nitrogen': tf.constant([int(data['nitrogen'])], dtype=tf.int64),
        'phosphorous': tf.constant([int(data['phosphorous'])], dtype=tf.int64),
        'potassium': tf.constant([int(data['potassium'])], dtype=tf.int64),
        'soil_type': tf.constant([data['soil_type']]),
        'temparature': tf.constant([int(data['temparature'])], dtype=tf.int64),
    }

    predictions = model.predict(input_data)
    predicted_label = tf.argmax(predictions, axis=1)[0].numpy()

    # Define the label mappings
    label_mappings = {
        0: "10-26-26",
        1: "14-35-14",
        2: "17-17-17",
        3: "20-20",
        4: "28-28",
        5: "DAP",
        6: "Urea"
    }

    # Print the predicted label
    if predicted_label in label_mappings:
        return jsonify({"predictions": label_mappings[predicted_label]})
    
    return jsonify({"predictions": 'Unknown label'})


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
