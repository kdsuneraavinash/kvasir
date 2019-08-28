from utils.printer import Printer

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from utils.modal_options import ModelOptions
from models.lib.inception import InceptionV4


class InceptionV4Model:
    def __init__(self, options: ModelOptions,  n_classes: int):
        self.name = "inception-v4"
        self.width = options.image_width
        self.height = options.image_height
        self.depth = 3
        self.learning_rate = options.learning_rate
        self.n_epochs = options.n_epochs
        self.batch_size = options.batch_size
        self.n_classes = n_classes

        self.model = None

    def build(self):
        """https://arxiv.org/abs/1602.07261"""
        
        if self.height != 299 or self.width != 299:
            Printer.error("ImageNet requires image sizes to be 299x299")
            exit()

        input_shape = (self.height, self.width,  self.depth)
        model = InceptionV4(num_classes=self.n_classes, dropout_prob=0.2)

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
