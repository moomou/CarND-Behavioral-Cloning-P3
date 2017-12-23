#!/usr/bin/env python
import keras
from keras.models import Model
from keras.layers import *
from keras.layers.merge import (add as l_add, multiply as l_multiply)


def Identity(name):
    return Lambda(lambda x: x, name=name)


def AttentionDense(size, name=None):
    _layer_cache = {}
    identity = Identity(name)

    def _AttentionDense(input_layer, name=name):
        attention_dim = size
        cache_name = 'att%d' % attention_dim
        if cache_name not in _layer_cache:
            _layer_cache[cache_name] = Dense(
                attention_dim, activation='softmax')
        attention_probs = _layer_cache[cache_name](input_layer)
        attention_mul = l_multiply([input_layer, attention_probs])
        attention_mul = identity(attention_mul)
        return attention_mul

    return _AttentionDense


def get_model(image_shape=(80, 160, 3)):
    fsize_pow = 5
    ksize = 5
    blk = 2

    x_out = x = Input(shape=image_shape, name='x')
    x_out = Cropping2D(cropping=((25, 10), (0, 0)))(x_out)
    x_out = Lambda(lambda x: (x / 255.0) - 0.5)(x_out)

    for i in range(blk):
        fsize = 2**(fsize_pow + i)
        x_out = Conv2D(fsize, ksize)(x_out)
        x_out = Conv2D(fsize, 1)(x_out)
        x_out = BatchNormalization()(x_out)
        x_out = Activation('relu')(x_out)
        x_out = Conv2D(fsize, ksize)(x_out)
        x_out = MaxPooling2D(2)(x_out)

    x_out = AttentionDense(2**6)(x_out)
    x_out = Flatten()(x_out)
    x_out = Dense(2**9)(x_out)
    x_out = Activation('relu')(x_out)
    x_out = Dropout(0.5)(x_out)
    x_out = Dense(2**8)(x_out)
    x_out = Activation('relu')(x_out)
    x_out = Dropout(0.5)(x_out)
    x_out = Dense(1)(x_out)

    m = Model(inputs=x, outputs=x_out)
    m.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.adam(), )

    m.summary()
    return m


if __name__ == '__main__':
    import fire
    fire.Fire()
