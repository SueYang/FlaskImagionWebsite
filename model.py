import os
from keras import applications
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import numpy as np

MODEL_WEIGHTS_NAME = "basic_model.h5"

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
        testobj = np.zeros((160, 160, 3))
        testobj = np.expand_dims(testobj, axis=0)
        custom_model.predict(testobj)
        return custom_model


def build_model():
    # If you want to specify input tensor
    input_tensor = Input(shape=(160, 160, 3))
    vgg_model = applications.VGG16(weights='imagenet',
                                   include_top=False,
                                   input_tensor=input_tensor)

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

    # Getting output tensor of the last VGG layer that we want to include
    x = layer_dict['block4_pool'].output

    # Stacking a new simple convolutional network on top of it
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)

    # my_model = pickle.load(open("file_predict.pkl", "rb"))
    # my_model = model_from_json(json_data)
    # Creating new model. Please note that this is NOT a Sequential() model.
    from keras.models import Model
    custom_model = Model(inputs=vgg_model.input, outputs=x)

    # Make sure that the pre-trained bottom layers are not trainable
    for layer in custom_model.layers[:15]:
        layer.trainable = False

    # Do not forget to compile it
    custom_model.compile(loss='categorical_crossentropy',
                         optimizer='rmsprop',
                         metrics=['accuracy'])
    return custom_model



