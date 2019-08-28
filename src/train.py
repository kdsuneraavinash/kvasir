import os
import random

import numpy as np
import tensorflow as tf

from models.simple_nn import SimpleNN
from models.vggnet import SmallVGGNet
from models.densenet import DenseNetModel
from models.inception import InceptionV4Model

from utils.dataset_loader import ImageDatasetLoader
from utils.modal_options import ModelOptions
from utils.printer import Printer
from utils.report import Report

if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    options = ModelOptions.ask()

    try:
        # Make random generation predictable
        random.seed(options.random_seed)
        np.random.seed(options.random_seed)

        dataset_loader = ImageDatasetLoader(options=options)
        dataset_loader.load(output_dir=options.output_dir)
        dataset_loader.encode_labels()
        classes = dataset_loader.class_names

        train_x, test_x, train_y, test_y = dataset_loader.split(
            test_percentage=options.test_percentage)

        if options.model_name == "SimpleNN":
            neuralNet: SimpleNN = SimpleNN(options=options, n_classes=len(classes))
        elif options.model_name == "DenseNet":
            neuralNet: DenseNetModel = DenseNetModel(options=options, n_classes=len(classes))
        elif options.model_name == "Inception":
            neuralNet: InceptionV4Model = InceptionV4Model(options=options, n_classes=len(classes))
        else:
            neuralNet: SmallVGGNet = SmallVGGNet(options=options, n_classes=len(classes))

        neuralNet.build()
        history = neuralNet.fit(train_x, train_y, test_x, test_y)

        report = Report(neuralNet, test_x, test_y,
                        classes, output_dir=options.output_dir)
        report.evaluation_report()
        report.chart(history)

        model_output_path = os.path.join(options.output_dir, '[{}]model.hdf5'.format(neuralNet.name))
        classes_output_path = os.path.join(options.output_dir, '[{}]classes.txt'.format(neuralNet.name))
        answers_output_path = os.path.join(options.output_dir, '[{}]options.json'.format(neuralNet.name))

        neuralNet.save(model_output_path)
        with open(classes_output_path, 'w') as fw:
            fw.write('\n'.join(classes))
        with open(answers_output_path, 'w') as fw:
            fw.write(str(options.to_json()).replace("'", '"'))

    except KeyboardInterrupt:
        Printer.error("Exiting (Ctrl+C)")
        exit()
