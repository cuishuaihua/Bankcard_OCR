from skimage import io, transform
import numpy as np
from keras.models import load_model
bank_dict = {0: 'False', 1: 'True'}
import cv2
import os

w = 200
h = 48
c = 3

def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img, (w, h))
    return np.expand_dims(np.asarray(img), axis=0)


model = load_model(os.getcwd()+'/comp/keras_vgg16_Bank_card.h5')

def predict(path):
    pre = model.predict(read_one_image(path))
    pre = np.argmax(pre,axis=1)
    return pre[0]

