from keras.models import Model
from keras.layers import Dense, Flatten
from skimage import io, transform
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.models import load_model
import cv2
path = r'./images/'  # 数据存放路径

w = 200
h = 48
c = 3
number_classification = 2


def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            # print('reading the images:%s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


data, label = read_img(path)
print("shape of data:", data.shape)
print("shape of label:", label.shape)

X_train, X_test, y_train, y_test = train_test_split(data, label,
                                                    test_size=0.2,
                                                    random_state=21)
y_train = np_utils.to_categorical(y_train, num_classes=number_classification)
y_test = np_utils.to_categorical(y_test, num_classes=number_classification)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(w, h, c))
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(number_classification, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)  # 指定模型的输入、输出
model.summary()  # 打印模型结构
print("=========\n", base_model.input)

for layer in base_model.layers:  # 设置禁止训练的层数
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=2, validation_data=(X_test, y_test))

print('\nTesting ------------')

loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
model.save('./keras_vgg16_Bank_card.h5')