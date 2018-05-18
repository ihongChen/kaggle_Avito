'''
this script copy from top kaggler CPMP

http://nbviewer.jupyter.org/github/jfpuget/LibFM_in_Keras/blob/master/keras_blog.ipynb
'''

from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dot, Reshape, Add, Subtract
from keras import objectives
from keras import backend as K
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2


import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

import os


def init_seeds(seed):
    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    np.random.seed(seed)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    rn.seed(seed)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    from keras import backend as K

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

    tf.set_random_seed(seed)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    return sess


def get_embed(x_input, x_size, k_latent):
    if x_size > 0:  # category
        embed = Embedding(x_size, k_latent, input_length=1,
                          embeddings_regularizer=l2(embedding_reg))(x_input)
        embed = Flatten()(embed)
    else:
        embed = Dense(k_latent, kernel_regularizer=l2(embedding_reg))(x_input)
    return embed


def build_model_1(X, f_size):
    dim_input = len(f_size)

    input_x = [Input(shape=(1,)) for i in range(dim_input)]

    biases = [get_embed(x, size, 1) for (x, size) in zip(input_x, f_size)]

    factors = [get_embed(x, size, k_latent)
               for (x, size) in zip(input_x, f_size)]

    s = Add()(factors)

    diffs = [Subtract()([s, x]) for x in factors]

    dots = [Dot(axes=1)([d, x]) for d, x in zip(diffs, factors)]

    x = Concatenate()(biases + dots)
    x = BatchNormalization()(x)
    output = Dense(1, activation='relu', kernel_regularizer=l2(kernel_reg))(x)
    model = Model(inputs=input_x, outputs=[output])
    opt = Adam(clipnorm=0.5)
    model.compile(optimizer=opt, loss='mean_squared_error')
    output_f = factors + biases
    model_features = Model(inputs=input_x, outputs=output_f)
    return model, model_features


if __name__ == '__main__':

    k_latent = 2
    embedding_reg = 0.0002
    kernel_reg = 0.1
