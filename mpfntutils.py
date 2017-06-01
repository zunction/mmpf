import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

import os
import timeit
from datetime import datetime




def load_data(dataset = '32-50K.npy'):
    """
    Loads the dataset.
    """
    return np.load(dataset)
