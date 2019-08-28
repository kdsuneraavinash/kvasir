from utils.printer import Printer

import inquirer
import os


class ModelOptions:
    def __init__(self):
        self.model_name: str = ""
        self.random_seed: int = 170081
        self.dataset_directory: str = "../images"
        self.image_extensions: set = {'jpg'}
        self.image_width: int = 64
        self.image_height: int = 64
        self.flatten: bool = False
        self.test_percentage: float = 0.2
        self.learning_rate: float = 0.01
        self.n_epochs: int = 50
        self.batch_size: int = 64
        self.output_dir: str = "../output"

    def to_json(self):
        json: dict = {'model_name': str(self.model_name),
                      'random_seed': str(self.random_seed),
                      'dataset_directory': str(self.dataset_directory),
                      'image_extensions': " ".join(self.image_extensions),
                      'image_width': str(self.image_width),
                      'image_height': str(self.image_height),
                      'flatten': str(bool(self.flatten)),
                      'test_percentage': str(self.test_percentage),
                      'learning_rate': str(self.learning_rate),
                      'n_epochs': str(self.n_epochs),
                      'batch_size': str(self.batch_size),
                      'output_dir': str(self.output_dir)}
        return json

    @staticmethod
    def from_json(json):
        options = ModelOptions()

        try:
            options.model_name = json['model_name']
            options.random_seed = int(json['random_seed'])
            options.dataset_directory = json['dataset_directory']
            options.image_extensions = set(json['image_extensions'].split())
            options.image_width = int(json['image_width'])
            options.image_height = int(json['image_height']) if json['image_height'] \
                else options.image_width
            options.flatten = (str(json['flatten']) == 'True')
            options.test_percentage = float(json['test_percentage'])
            options.learning_rate = float(json['learning_rate'])
            options.n_epochs = int(json['n_epochs'])
            options.batch_size = int(json['batch_size'])
            options.output_dir = json['output_dir']
        except ValueError:
            Printer.error("Input data parsing failed.")

        return options

    @staticmethod
    def ask():
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
            inquirer.List('model_name', message='What model to use?', 
                         choices=['SimpleNN', 'Small VGGNet', 'DenseNet', 'Inception']),
            inquirer.Text('random_seed', message='Set the random seed',
                          default='170081', validate=is_digit_and_positive),
            inquirer.Text('dataset_directory', message='What is the dataset directory?',
                          default='../images', validate=is_directory),
            inquirer.Text('image_extensions', message='What are image  extensions?(Separate with space)',
                          default='jpg', validate=lambda _, x: len(x) > 1),
            inquirer.Text('image_width', message='What is the image height? (Use 299 for Inception)',
                          default='64', validate=is_digit_and_positive),
            inquirer.Text('image_height', message='What is the image weight? (Leave empty to use width)',
                          validate=lambda _, x: len(x) == 0 or is_digit_and_positive(_, x)),
            inquirer.Confirm('flatten', message='Do images need to be flattened before feeding to the network?'),
            inquirer.Text('test_percentage', message='How much to split for test data?',
                          default='0.15', validate=float_and_between_zero_one),
            inquirer.Text('learning_rate', message='Set learning rate',
                          default='0.01', validate=float_and_between_zero_one),
            inquirer.Text('n_epochs', message='Set number of epochs',
                          default='50', validate=is_digit_and_bigger_than_one),
            inquirer.Text('batch_size', message='Set batch size',
                          default='64', validate=is_digit_and_bigger_than_one),
            inquirer.Text('output_dir', message='What is the output directory?',
                          default='../output', validate=is_directory)
        ]

        try:
            answers = inquirer.prompt(questions)
            return ModelOptions.from_json(answers)
        except KeyboardInterrupt:
            Printer.error("User exit (Ctrl+C)")
        return ModelOptions()

    def get_dataset_cache_name(self):
        name = "dataset{}-{}x{}.cache".format("[flat]" if self.flatten else "", self.image_width, self.image_height)
        return name
