from utils.printer import Printer

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


class Report:
    def __init__(self, neural_net, sample_x, sample_y, class_names, output_dir='output'):
        self.neural_net = neural_net
        self.sample_x = sample_x
        self.sample_y = sample_y
        self.classes = class_names
        self.output_dir = output_dir

    def evaluation_report(self):
        """Generate report on test data"""

        predictions = self.neural_net.predict(self.sample_x)

        y_true = self.sample_y.argmax(axis=1)
        y_pred = predictions.argmax(axis=1)

        report = classification_report(y_true, y_pred, target_names=self.classes)

        report_output_path = os.path.join(self.output_dir,
                                          f'[{self.neural_net.name}]evaluation.txt')

        with open(report_output_path, 'w') as fw:
            fw.write(str(report))

        Printer.default(report)

    def chart(self, history):
        """Generate, show and save a plot on accurcy and loss by epoch number"""

        x_axis_values = np.arange(0, self.neural_net.n_epochs)
        plt.style.use('seaborn-white')
        plt.figure()

        plt.plot(x_axis_values, history.history['loss'], label='Loss(Train)')
        plt.plot(x_axis_values,
                 history.history['val_loss'], label='Loss(Test)')
        plt.plot(x_axis_values,
                 history.history['acc'], label='Accuracy(Train)')
        plt.plot(x_axis_values,
                 history.history['val_acc'], label='Accuracy(Test)')

        plt.title("Training Loss and Accuracy ({})".format(self.neural_net.name))
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss/Accuracy")
        plt.legend()

        chart_output_path = os.path.join(self.output_dir,
                                         f'[{self.neural_net.name}]chart')
        plt.savefig(chart_output_path)
        plt.show()
