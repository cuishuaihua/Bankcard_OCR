from skimage import io, transform
import numpy as np
from keras.models import load_model
bank_dict = {0: 'False', 1: 'True'}
import os
w = 200
h = 48
c = 3

def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img, (w, h))
    return np.expand_dims(np.asarray(img), axis=0)
model = load_model('./keras_vgg16_Bank_card.h5')

#./images/1/27.jpg
path = 'images/0/'
list = os.listdir(path)


for i in list:

    name = path + i

    pre = model.predict(read_one_image(name))
    pre = np.argmax(pre,axis =1)

    for j in range(len(pre)):
        print("Bank prediction:" + bank_dict[pre[j]])


'''
while True:
    prepath = 'images/0/'
    data=prepath+input("")
    pre = model.predict(read_one_image(data))
    pre = np.argmax(pre, axis=1)
    for i in range(len(pre)):
        print("Bank prediction:" + bank_dict[pre[i]])
'''
