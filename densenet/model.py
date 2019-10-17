#-*- coding:utf-8 -*-
import os
import numpy as np
from imp import reload
from PIL import Image, ImageOps

from keras.layers import Input
from keras.models import Model

from . import keys
from . import densenet

'''
    文本识别模块
'''

reload(densenet)

characters = keys.alphabet[:]
characters = characters[1:] + u'卍'
nclass = len(characters)

#构建模型
input = Input(shape=(32, None, 1), name='the_input')
y_pred= densenet.dense_cnn(input, nclass)
basemodel = Model(inputs=input, outputs=y_pred)


modelPath = os.getcwd()[:-5]+'/densenet/models/weights_densenet.h5'
if os.path.exists(modelPath):
    basemodel.load_weights(modelPath)

def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]

    # print("test before:",pred_text)

    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(characters[pred_text[i]])
    return u''.join(char_list)

def predict(img):

    width, height = img.size[0], img.size[1]

    scale = height * 1.0 / 32
    if scale>1.0:
        width = int(width / abs(scale-0.45))
    else:
        width = int(width / abs(scale))
    img = img.resize([width, 32], Image.ANTIALIAS)


    img = np.array(img).astype(np.float32) / 255.0 - 0.5
    
    X = img.reshape([1, 32, width, 1])
    
    y_pred = basemodel.predict(X)
    y_pred = y_pred[:, :, :]

    # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
    # out = u''.join([characters[x] for x in out[0]])
    out = decode(y_pred)

    # print("out",out)
    return out
