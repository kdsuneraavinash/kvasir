from utils.printer import Printer

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense

from utils.modal_options import ModelOptions


class SimpleNN:
    def __init__(self, options: ModelOptions, n_classes: int):
        self.name = 'simple_nn'
        self.width = options.image_width
        self.height = options.image_height
        self.learning_rate = options.learning_rate
        self.n_epochs = options.n_epochs
        self.batch_size = options.batch_size
        self.n_classes = n_classes
        self.model = None

        if not options.flatten:
            Printer.error("Images has to be flattened before sending to Small VGGNet")
            exit()

    def build(self):
        """
        (Dense => Sigmoid) => (Dense => Sigmoid) => (Dense => Sigmoid) => (Dense => Sigmoid)
        """

        input_shape = (self.width * self.height * 3,)
        output_nodes = self.n_classes

        model = Sequential()
        model.add(Dense(1024, input_shape=input_shape, activation='sigmoid'))
        model.add(Dense(512, activation='sigmoid'))
        model.add(Dense(output_nodes, activation='sigmoid'))

        optimizer = SGD(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
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
