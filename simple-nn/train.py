import os
import cv2
import random
import inquirer
import numpy as np
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.models import Sequential
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


class Print:
    @staticmethod
    def information(message):
        print(f"[INFO] {message}")

    @staticmethod
    def error(message):
        print(f"[WARN] {message}")


###############################################################################
#               Loading Dataset
###############################################################################


def preprocess_image(image_path, width, height):
    """ Read and process image given by `image_path` and
    returns the image with size `width*height*3` """

    image = cv2.imread(image_path)
    image = cv2.resize(image, (width, height))
    image = image.flatten()
    return image


def load_image_paths(base_path, image_extensions={'jpg'}):
    """ Return a list of image files inside the `base_path` """

    image_paths = []
    for (dir_path, _, file_names) in os.walk(base_path):
        for file_name in file_names:
            if '.' in file_name:
                extension = file_name.split('.')[-1]
                if extension in image_extensions:
                    if '.ipynb' in dir_path:
                        continue
                    image_paths.append(os.path.join(dir_path, file_name))

    random.shuffle(image_paths)

    Print.information(f"Found {len(image_paths)} images")
    return image_paths


def load_image_dataset(dataset_path, image_width=32, image_height=32,
                       notification_milestone=250, image_extensions={'jpg'}):
    """Load dataset from `dataset_path` into memory and return
    a tuple of (images, labels) where `image[i]` is the i'th preprocessed image
    normalized into 0-1 range
    and `labels[i]` is the label of i'th image.

    `dataset_path` must have directories representing labels and
    images inside each directory."""

    images = []
    labels = []

    image_paths = load_image_paths(dataset_directory,
                                   image_extensions=image_extensions)

    Print.information("Started loading dataset")

    n_images = len(image_paths)
    for image_path_ind in range(n_images):
        image_path = image_paths[image_path_ind]

        image = preprocess_image(image_path, image_width, image_height)
        label = image_path.split(os.path.sep)[-2]

        images.append(image)
        labels.append(label)

        if image_path_ind % notification_milestone == 0:
            Print.information(f"Loaded {image_path_ind}/{n_images} images")

    Print.information("Dataset loaded into memory")

    images = np.array(images, dtype='float') / 255
    labels = np.array(labels)

    return images, labels


###############################################################################
#               Split Dataset
###############################################################################


def prepare_dataset(samples, labels, test_size=0.25):
    """ Split dataset into test and train samples.
    This also one hot encodes labels."""

    x, test_x, y, test_y = train_test_split(samples, labels,
                                            test_size=test_split_percentage,
                                            shuffle=True)
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(y)

    n_classes = label_binarizer.classes_
    if n_classes != len(set(test_y)):
        Print.error("Test set does not contain one of each class.")

    y = label_binarizer.transform(y)
    test_y = label_binarizer.transform(test_y)

    return label_binarizer.classes_, x, test_x, y, test_y


###############################################################################
#               Modeling and training Neural Network
###############################################################################


def define_model(sample_size, n_classes, learning_rate=0.01):

    input_nodes = sample_size
    output_nodes = n_classes

    model = Sequential()
    model.add(Dense(1024, input_shape=(input_nodes, ), activation='sigmoid'))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(output_nodes, activation='sigmoid'))

    optimizer = SGD(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def train(train_x, test_x, train_y, test_y, sample_size, n_classes,
          learning_rate=0.01, batch_size=32, n_epochs=50):
    """Defines and compiles the required model (Simple Neural Network)
    - input_size > 1024(sigmoid) >  512(sigmoid) > output(sigmoid)
    - loss = categorical_crossentropy
    - optimizer = Stochastic Gradient Descent

    After defining, trains the model on given data and returns history object"""

    model = define_model(sample_size, n_classes,
                         learning_rate=learning_rate)

    Print.information("Training network")
    history = model.fit(x=train_x,
                        y=train_y,
                        batch_size=batch_size,
                        epochs=n_epochs,
                        validation_data=(test_x, test_y))
    Print.information("Training completed")

    return model, history


###############################################################################
#               Evaluate Model
###############################################################################


def get_evaluation_report(model, test_x, test_y, classes, batch_size=32):
    predictions = model.predict(test_x, batch_size=batch_size)

    y_true = test_y.argmax(axis=1)
    y_pred = predictions.argmax(axis=1)

    report = classification_report(y_true, y_pred, target_names=classes)
    return report


def show_and_save_chart(history, classes, save_path='evaluation', n_epochs=50):
    x_axis_values = np.arange(0, n_epochs)
    plt.style.use('seaborn-white')
    plt.figure()

    plt.plot(x_axis_values, history.history['loss'], label='Loss(Train)')
    plt.plot(x_axis_values, history.history['val_loss'], label='Loss(Test)')
    plt.plot(x_axis_values, history.history['acc'], label='Accuracy(Train)')
    plt.plot(x_axis_values, history.history['val_acc'], label='Accuracy(Test)')

    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss/Accuracy")
    plt.legend()

    plt.savefig(save_path)
    plt.show()


###############################################################################
#               Save Model and Evaluations
###############################################################################

def save(model, report, classes, answers, output_dir='output'):
    model_output_path = os.path.join(output_dir, 'model.hdf')
    classes_output_path = os.path.join(output_dir, 'classes.txt')
    report_output_path = os.path.join(output_dir, 'evaluation.txt')
    answers_output_path =  'options.json'

    model.save(model_output_path)
    with open(classes_output_path, 'w') as fw:
        fw.write('\n'.join(classes))
    with open(report_output_path, 'w') as fw:
        fw.write(str(report))
    with open(answers_output_path, 'w') as fw:
        fw.write(str(answers).replace("'", '"'))


###############################################################################
#               Main
###############################################################################

if __name__ == "__main__":
    def is_digit_and_positive(_, x):
        return x.isdigit() and 0 < int(x)

    def is_digit_and_bigger_than_one(_, x):
        return x.isdigit() and 1 < int(x)

    def float_and_between_zero_one(_, x):
        try:
            x = float(x)
            return 0 < x < 1
        except ValueError:
            return False

    def is_directory(_, x):
        return os.path.isdir(x)

    questions = [
        inquirer.Text('random_seed', message='Set the random seed',
                      default='170081', validate=is_digit_and_positive),
        inquirer.Text('dataset_directory', message='What is the dataset directory?',
                      default='images',  validate=is_directory),
        inquirer.Text('image_extensions', message='What are image  extensions?(Seperate with space)',
                      default='jpg',  validate=lambda _, x: len(x) > 1),
        inquirer.Text('image_width', message='What is the image height?',
                      default='32',  validate=is_digit_and_positive),
        inquirer.Text('image_height', message='What is the image weight? (Leave empty to use width)',
                      validate=lambda _, x: len(x) == 0 or is_digit_and_positive(_, x)),
        inquirer.Text('notif_gap', message='Set how often to notify data file opening?',
                      default='250', validate=is_digit_and_positive),
        inquirer.Text('test_split_percentage', message='How much to split for test data?',
                      default='0.25', validate=float_and_between_zero_one),
        inquirer.Text('learning_rate', message='Set learning rate',
                      default='0.01', validate=float_and_between_zero_one),
        inquirer.Text('n_epochs', message='Set number of epochs',
                      default='50',  validate=is_digit_and_bigger_than_one),
        inquirer.Text('batch_size', message='Set batch size',
                      default='32',  validate=is_digit_and_bigger_than_one),
        inquirer.Text('output_dir', message='What is the output directory?',
                      default='output',  validate=is_directory)
    ]
    answers = inquirer.prompt(questions)

    try:
        random_seed = int(answers['random_seed'])
        dataset_directory = answers['dataset_directory']
        image_extensions = set(answers['image_extensions'].split())
        image_width = int(answers['image_width'])
        image_height = image_width if len(answers['image_height']) == 0  \
            else int(answers['image_height'])
        notif_gap = int(answers['notif_gap'])
        test_split_percentage = float(answers['test_split_percentage'])
        learning_rate = float(answers['learning_rate'])
        n_epochs = int(answers['n_epochs'])
        batch_size = int(answers['batch_size'])
        output_dir = answers['output_dir']
    except:
        Print.error("Input data invalid.")
        exit()

    try:
        # Make random generation predictable
        random.seed(random_seed)
        np.random.seed(random_seed)

        samples, labels = load_image_dataset(dataset_directory,
                                             image_width=image_width, image_height=image_height,
                                             notification_milestone=notif_gap,
                                             image_extensions=image_extensions)
        classes, train_x, test_x, train_y, test_y = prepare_dataset(samples, labels,
                                                                    test_size=test_split_percentage)
        model, history = train(train_x, test_x, train_y, test_y,
                               image_width*image_height*3, len(classes),
                               batch_size=batch_size, n_epochs=n_epochs)

        report = get_evaluation_report(model, test_x, test_y, classes,
                                       batch_size=batch_size)
        print(report)
        show_and_save_chart(history, classes, save_path=os.path.join(output_dir, 'evaluation'),
                            n_epochs=n_epochs)
        save(model, report, classes, answers=answers, output_dir=output_dir)
    except KeyboardInterrupt:
        Print.error("Exiting.")
        exit()
