from flask import Flask, render_template, request
import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import joblib
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('fracture.html')


@app.route('/upload', methods=['POST'])
def upload():
    image = request.files['image']
    # Do something with the uploaded image
    # For example, you can save it to disk
    image.save('uploaded_image.jpg')

    img = cv2.imread('uploaded_image.jpg')

    resize = tf.image.resize(img, (256, 256))
    # plt.imshow(resize.numpy().astype(int))
    # plt.show()
    knn_from_joblib = joblib.load('fracture.pkl')
    yhat = knn_from_joblib.predict(np.expand_dims(resize/255, 0))
    print(yhat)
    if yhat > 0.5:
        # print('Tumor Present')
        a = "Fracture Absent"
    else:
        # print('No tumor :)')
        a = "Fracture Present"

    return render_template('result.html', output=a)


if __name__ == '__main__':
    app.run()
