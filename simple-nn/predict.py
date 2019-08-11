import cv2
import os
import json
import random
import inquirer
import numpy as np
from keras.models import load_model
from train import preprocess_image, load_image_paths, Print

MAX_PREDICTIONS = 100


def load_images(images_directory, image_width=32, image_height=32,
                image_extensions={'jpg'}):
    """Load images from `images_directory` into memory and return
    `images` where `image[i]` is the i'th preprocessed image
    normalized into 0-1 range."""

    images = []

    if os.path.isfile(images_directory):
        if '.' in images_directory:
            ext = images_directory.split('.')[-1]
            if ext in image_extensions:
                # directory is an image
                image_paths = [images_directory]
    else:
        image_paths = load_image_paths(images_directory,
                                       image_extensions=image_extensions)

    Print.information("Started loading images")

    n_images = len(image_paths)
    for image_path_ind in range(n_images):
        if image_path_ind >= MAX_PREDICTIONS:
            break
        image_path = image_paths[image_path_ind]
        image = preprocess_image(image_path, image_width, image_height)
        images.append(image)
    Print.information("Images loaded into memory")

    images = np.array(images, dtype='float') / 255
    return images, image_paths


if __name__ == "__main__":
    questions = [
        inquirer.Text('options', message='What is the options file path?',
                      default='options.json',  validate=lambda _, x: os.path.isfile(x)),
        inquirer.Text('images_directory', message='What path of the folder of files to predict?(If single file, enter path)',
                       validate=lambda _, x: os.path.isfile(x) or os.path.isdir(x)),
        inquirer.Text('image_extensions', message='What are image  extensions?(Seperate with space)',
                      default='jpg',  validate=lambda _, x: len(x) > 1)
    ]
    answers = inquirer.prompt(questions)

    try:
        answers_output_path = answers['options']
        image_extensions = set(answers['image_extensions'].split())
        images_directory = answers['images_directory']
    except:
        Print.error("Input data invalid.")
        exit()

    with open(answers_output_path, 'r') as fr:
        answers = json.loads(fr.read().strip())
    random_seed = int(answers['random_seed'])
    image_width = int(answers['image_width'])
    image_height = image_width if len(answers['image_height']) == 0  \
        else int(answers['image_height'])
    learning_rate = float(answers['learning_rate'])
    output_dir = answers['output_dir']

    # Make random generation predictable
    random.seed(random_seed)
    np.random.seed(random_seed)

    model_output_path = os.path.join(output_dir, 'model.hdf')
    classes_output_path = os.path.join(output_dir, 'classes.txt')
    images, image_paths = load_images(images_directory,
                                      image_width=image_width, image_height=image_height,
                                      image_extensions=image_extensions)

    model = load_model(model_output_path)
    with open(classes_output_path, 'r') as fr:
        classes = fr.read().split('\n')
        classes = list(map(str.strip, classes))
    predictions = model.predict(images)

    print()
    for ind in range(len(images)):
        prediction = predictions[ind]
        image_path = image_paths[ind]

        pred_i = prediction.argmax(axis=0)
        predicted_label = classes[pred_i]
        print(
            f'{image_path}\t{predicted_label:<22}\t{prediction[pred_i] * 100:.2f}%')
