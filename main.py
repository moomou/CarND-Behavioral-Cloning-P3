#!/usr/bin/env python
import os

import keras

from data import read_data
from model import get_model

monitor = 'val_loss'

if __name__ == '__main__':
    train_gen, val_gen, train_steps, val_steps = read_data()
    model = get_model()

    if os.environ.get('WEIGHT'):
        model.load_weights(os.environ.get('WEIGHT'))

    callbacks = [
        # keras.callbacks.TensorBoard(
        # './log_dir', histogram_freq=2, write_grads=True),
        keras.callbacks.ModelCheckpoint(
            os.path.join('./ckpts',
                         'chkpt.{epoch:05d}.{%s:.5f}.hdf5' % monitor),
            mode='min',
            monitor=monitor)
    ]

    model.fit_generator(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        epochs=15)
