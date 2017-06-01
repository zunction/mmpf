import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

import os
import timeit
from datetime import datetime



def load_data(dataset = '32-50K.npy', borrow = True):
    """
    Loads the dataset. With Theano consideration when borrow True.
    """
    if not borrow:
        dataset = np.load(dataset)
    else:
        data = np.load(dataset)
        dataset = theano.shared(np.asarray(data, dtype = theano.config.floatX), borrow = borrow)

    return dataset

# def load_data(dataset = '32-50K.npy'):
#     """
#     Loads the dataset.
#     """
#     return np.load(dataset)
