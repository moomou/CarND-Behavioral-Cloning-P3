import os
import random

import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

DATA_ROOT = './data'
DATA_FOLDERS = [
    os.path.join(DATA_ROOT, 'new'),
    os.path.join(DATA_ROOT, 't1'),
    os.path.join(DATA_ROOT, 't2'),
]

OFFSET = 0.35


def augment_images(data):
    # augments the data by adding a tuple (image_index, flip or not) to the
    # data tuple
    new_x = []
    for x in data:
        new_x.append((0, 0) + x)
        new_x.append((0, 1) + x)
        # use left
        new_x.append((1, 0) + x)
        new_x.append((1, 1) + x)
        # use right
        new_x.append((2, 0) + x)
        new_x.append((2, 1) + x)

    return new_x


def proc_img(img):
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            correction = 0.
            for img_indx, aug, root, batch_sample in batch_samples:
                name = os.path.join(root, 'IMG/',
                                    batch_sample[img_indx].split('/')[-1])
                if img_indx == 0:
                    correction = 0
                elif img_indx == 1:
                    correction = OFFSET
                elif img_indx == 2:
                    correction = -1 * OFFSET

                img = cv2.imread(name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = proc_img(img)
                angle = float(batch_sample[3]) + correction

                if aug:
                    img = cv2.flip(img, 0)
                    angle *= -1

                images.append(img)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield X_train, y_train


def read_data():
    samples = []
    for f in DATA_FOLDERS:
        with open(os.path.join(f, './driving_log.csv')) as csvfile:
            reader = csv.reader(csvfile)
            samples.extend([(f, line) for line in reader])

    aug_samples = augment_images(samples)
    aug_samples = sklearn.utils.shuffle(aug_samples)

    train_samples, validation_samples = train_test_split(
        aug_samples, test_size=0.1)

    print('Total train samples::', len(train_samples))
    print('Total test samples::', len(validation_samples))

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    return train_generator, validation_generator, len(
        train_samples) // 32, len(validation_samples) // 32
