import os
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
import numpy as np

MODEL_WEIGHTS_NAME = "model_regress.h5"

class Model(object):
    def __init__(self, app):
        self.app = app

    def model(self):
        return self.model

    def model_weights_path(self):
        return os.path.join(self.app.config['MODEL_WEIGHTS_FOLDER'], MODEL_WEIGHTS_NAME)

    def get_model(self):
        custom_model = build_model()
        weights_path = self.model_weights_path()
        custom_model.load_weights(weights_path)
        # add below code to make the model run prediction once. Otherwise there would be error on GCE
        # https://zhuanlan.zhihu.com/p/27101000
        testobj = np.zeros((100, 100, 3))
        testobj = np.expand_dims(testobj, axis=0)
        custom_model.predict(testobj)
        return custom_model


def build_model():
    img_rows = 100
    img_cols = 100

    input_shape = (img_rows, img_cols, 3)

    custom_model = Sequential()
    custom_model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    custom_model.add(Activation('relu'))
    custom_model.add(Conv2D(32, (3, 3)))
    custom_model.add(Activation('relu'))
    custom_model.add(MaxPooling2D(pool_size=(2, 2)))
    custom_model.add(Dropout(0.25))

    custom_model.add(Conv2D(64, (3, 3), padding='same'))
    custom_model.add(Activation('relu'))
    custom_model.add(Conv2D(64, (3, 3)))
    custom_model.add(Activation('relu'))
    custom_model.add(MaxPooling2D(pool_size=(2, 2)))
    custom_model.add(Dropout(0.25))

    custom_model.add(Flatten())
    custom_model.add(Dense(512))
    custom_model.add(Activation('relu'))
    custom_model.add(Dropout(0.5))
    custom_model.add(Dense(1))
    custom_model.add(Activation('linear'))

    # Custom Optimizer
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=1e-6)

    # Do not forget to compile it
    custom_model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    return custom_model



