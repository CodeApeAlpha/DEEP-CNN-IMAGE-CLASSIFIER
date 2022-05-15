

import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image

from matplotlib import pyplot as plt

# Created model
new_model = load_model('./Models/imageclassifier.h5')
# Load image
img = cv2.imread('Test/FB_IMG_1590905044280.jpg')


# Preprocessing
img = image.load_img('./Test/FB_IMG_1590905044280.jpg', target_size=(256, 256))
# img = image.load_img('../FB_IMG_1590905044280.jpg', target_size=(img_width, img_height))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
yhat = new_model.predict(images)

print(yhat[0])

# if yhat > 0.5:
#     print(f'Predicted class is Sad')
# else:
#     print(f'Predicted class is Happy')

