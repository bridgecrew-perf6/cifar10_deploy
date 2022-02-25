from flask import Flask, render_template, request

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model_path = 'models/cifar10_model.h5'
model = tf.keras.models.load_model(model_path)


# model._make_predict_function()


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path)
    image = img_to_array(image)
    image = image.reshape((1, 32, 32, 3))
    image = image / 255

    pred = model.predict(image)
    pred = np.argmax(pred)
    # return pred

    return render_template('index.html', prediction=pred)


if __name__ == "__main__":
    app.run(port=3000, debug=True)
