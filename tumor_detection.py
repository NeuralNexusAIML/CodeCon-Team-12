import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import joblib
img = cv2.imread('normalbrain.png')
resize = tf.image.resize(img, (256, 256))
# plt.imshow(resize.numpy().astype(int))
# plt.show()
knn_from_joblib = joblib.load('tumor_detection.pkl')
yhat = knn_from_joblib.predict(np.expand_dims(resize/255, 0))
print(yhat)
if yhat > 0.5:
    print('Tumor Present')
else:
    print('No tumor :)')
