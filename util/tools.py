import numpy as np
import os
import pickle
import random
import tensorflow as tf

from tensorflow.python.keras import backend


SEED = 0


def init_environment():
    """To init the environment and keep reproducible results"""
    # Clearing previous TF sessions
    backend.clear_session()

    # Fixing the seed for random number generators
    os.environ['PYTHONHASHSEED'] = str(2)
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    print('Initializing TF Session and Seed')


def save_object(obj, filename):
    """To save objects that can be loaded later"""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """Get back the saved object"""
    with open(filename, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model
