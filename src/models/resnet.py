from utils.printer import Printer

from utils.modal_options import ModelOptions

import keras
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers


class ResNet:
    def __init__(self, options: ModelOptions, n_classes: int):
        self.name = 'resnet34'
        self.width = options.image_width
        self.height = options.image_height
        self.learning_rate = options.learning_rate
        self.n_epochs = options.n_epochs
        self.batch_size = options.batch_size
        self.n_classes = n_classes
        self.model = None

    def build(self):
        input_shape = (self.width * self.height * 3,)
        output_nodes = self.n_classes

        restnet = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
        output = restnet.layers[-1].output
        output = keras.layers.Flatten()(output)

        restnet = Model(restnet.input, output=output)

        for layer in restnet.layers:
            layer.trainable = False

        model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        model.add(Dense(1024, input_shape=input_shape, activation='sigmoid'))
        model.add(Dense(512, activation='sigmoid'))
        model.add(Dense(output_nodes, activation='sigmoid'))

        model = Sequential()
        model.add(restnet)
        model.add(Dense(512, activation='relu', input_dim=input_shape))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])

        self.model = model

    def fit(self, train_x, train_y, test_x, test_y):
        if self.model is None:
            Printer.warning("Model was automatically built when fitting.")
            self.build()

        history = self.model.fit(x=train_x,
                                 y=train_y,
                                 batch_size=self.batch_size,
                                 epochs=self.n_epochs,
                                 validation_data=(test_x, test_y))
        return history

    def predict(self, x):
        if self.model is None:
            Printer.error("Model not built nor trained yet. Cannot predict.")
        return self.model.predict(x, batch_size=self.batch_size)

    def save(self, model_output_path: str):
        self.model.save(model_output_path)
        Printer.information(f"Model saved as {model_output_path}")
