from utils.printer import Printer

from keras.optimizers import SGD
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense

from utils.modal_options import ModelOptions


class SmallVGGNet:
    def __init__(self, options: ModelOptions,  n_classes: int):
        self.name = "small_vggnet"
        self.width = options.image_width
        self.height = options.image_height
        self.depth = 3
        self.learning_rate = options.learning_rate
        self.n_epochs = options.n_epochs
        self.batch_size = options.batch_size
        self.n_classes = n_classes

        self.model = None

        if options.flatten:
            Printer.error("Images has to be flattened before sending to Small VGGNet")
            exit()

    def build(self):
        """
        (Conv => Relu) +=> Pool <>
        (Conv => Relu) +=> (Conv => Relu) +=> Pool <>
        (Conv => Relu) +=> (Conv => Relu) +=> (Conv => Relu) +=> Pool <>

        Conv2D = Convolution input layer
        Activation(relu) = ReLU (Rectified Linear Unit) activation function
        BatchNormalization = Normalize the activations of a given input volume (reduces epochs to train)
        MaxPooling2D = Reduce spatial size (downsampling)
        Dropout = Disconnecting random neurons between layers (reduce overfitting)
        """
        model = Sequential()
        input_shape = (self.height, self.width,  self.depth)
        channel_dimension = -1  # Specific for tensorflow - channels_last

        model.add(Conv2D(32, (3, 3), padding="same", activation="relu",
                         input_shape=input_shape))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=channel_dimension))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(self.n_classes))
        model.add(Activation("softmax"))

        optimizer = SGD(lr=self.learning_rate,
                        decay=self.learning_rate/self.n_epochs)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])

        self.model = model

    def fit(self, train_x, train_y, test_x, test_y):
        if self.model is None:
            Printer.warning("Model was automatically built when fitting.")
            self.build()

        data_augmenter = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                                            height_shift_range=0.1, shear_range=0.2,
                                            zoom_range=0.2, horizontal_flip=True,
                                            fill_mode='nearest')

        steps_per_epoch = len(train_x)//self.batch_size
        history = self.model.fit_generator(data_augmenter.flow(train_x, train_y, batch_size=self.batch_size),
                                           validation_data=(test_x, test_y),
                                           steps_per_epoch=steps_per_epoch,
                                           epochs=self.n_epochs)
        return history

    def predict(self, x):
        if self.model is None:
            Printer.error("Model not built nor trained yet. Cannot predict.")
        return self.model.predict(x, batch_size=self.batch_size)

    def save(self, model_output_path: str):
        self.model.save(model_output_path)
        Printer.information(f"Model saved as {model_output_path}")
