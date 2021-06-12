import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import os

import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from skimage.transform import resize

app = Flask(__name__)

model = load_model('models/bs_xception_model.h5')


@app.route("/", methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')


@app.route('/prediction/<filename>')
def prediction(filename):
    file_path = os.path.join('uploads', filename)

    my_image = plt.imread(file_path)
    my_image_re = resize(my_image, (299, 299, 3))

    probabilities = model.predict(np.array([my_image_re, ]))[0, :]
    probs_rounded = list(map(convert_probabilities, probabilities))

    number_to_class = ['bee', 'wasp', 'other insect', 'other thing']
    index = np.argsort(probs_rounded)
    predictions = {
        "class1": number_to_class[index[3]],
        "class2": number_to_class[index[2]],
        "class3": number_to_class[index[1]],
        "class4": number_to_class[index[0]],
        "prob1": probs_rounded[index[3]],
        "prob2": probs_rounded[index[2]],
        "prob3": probs_rounded[index[1]],
        "prob4": probs_rounded[index[0]],
    }

    os.remove(file_path)
    return render_template('predict.html', predictions=predictions)


def convert_probabilities(prob):
    return round(prob * 100, 2)


if __name__ == "__main__":
    app.run(host="localhost", port=8080, debug=True)
