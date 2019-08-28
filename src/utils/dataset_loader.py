from utils.modal_options import ModelOptions
from utils.printer import Printer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pickle
import random
import cv2
import sys
import os


class ImageDatasetLoader:
    def __init__(self, options: ModelOptions):
        self.base_path: str = options.dataset_directory
        self.flatten = options.flatten
        self.width: int = options.image_width
        self.height: int = options.image_height
        self.image_extensions: set = options.image_extensions
        self.cache_file_name = options.get_dataset_cache_name()

        self.images: list = []
        self.labels: list = []
        self.encoded_labels: list = []

        self.class_names: list = []

    def _process_image(self, image_path: str):
        """Read and process image given by `image_path` and
        returns the image with size `width*height*3`"""

        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.width, self.height))
        image = np.array(image, dtype='float') / 255
        if self.flatten:
            image = image.flatten()
        return image

    def _get_image_paths(self):
        """Return a list of image files inside the `base_path` """

        image_paths = []
        for (dir_path, _, file_names) in os.walk(self.base_path):
            for file_name in file_names:
                if os.extsep not in file_name:
                    Printer.warning("Files without extension found: {}"
                                    .format(file_name))
                    continue
                extension = file_name.split(os.extsep)[-1]
                if extension not in self.image_extensions:
                    Printer.warning("Non-image files found: {}"
                                    .format(file_name))
                    continue
                if '.ipynb' in dir_path:
                    Printer.warning("IPyNb caches found: {}"
                                    .format(file_name))
                    continue
                image_paths.append(os.path.join(dir_path, file_name))

        random.shuffle(image_paths)

        Printer.information(f"Found {len(image_paths)} images")
        return image_paths

    def load(self, output_dir="output"):
        """Load dataset from `dataset_path` into memory and return
        a tuple of (images, labels) where `image[i]` is the i'th preprocessed image
        normalized into 0-1 range
        and `labels[i]` is the label of i'th image.

        `dataset_path` must have directories representing labels and
        images inside each directory."""

        cache_file = os.path.join(output_dir, self.cache_file_name)
        Printer.information("Searching for cache file: " + cache_file)

        if os.path.exists(cache_file):
            Printer.information("Cache file found. Loading from cache.")
            with open(cache_file, 'rb') as fr:
                images, labels = pickle.load(fr)
        else:
            Printer.warning("Cache file not found")
            Printer.information("Started loading dataset")

            images = []
            labels = []

            image_paths = self._get_image_paths()

            for ind, image_path in enumerate(image_paths):
                image = self._process_image(image_path)
                label = image_path.split(os.path.sep)[-2]

                images.append(image)
                labels.append(label)

                Printer.processing(f"Loaded {ind}/{len(image_paths)} images.")

            Printer.end_processing()

            images = np.array(images, dtype='float')
            labels = np.array(labels)

            if not sys.getsizeof(images) > 1024*1024*1024:
                with open(cache_file, 'wb') as fw:
                    pickle.dump((images, labels), fw)

        Printer.information("Dataset loaded into memory")
        self.images = images
        self.labels = labels

    def encode_labels(self):
        label_binarizer = LabelBinarizer()
        self.encoded_labels = label_binarizer.fit_transform(self.labels)
        self.class_names = label_binarizer.classes_

        Printer.information("Labels Found: " + (", ".join(self.class_names)))

    def split(self, test_percentage=0.25):
        """ Split dataset into test and train samples."""
        return train_test_split(self.images, self.encoded_labels,
                                test_size=test_percentage,
                                stratify=self.encoded_labels)
