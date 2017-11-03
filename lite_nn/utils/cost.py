'''
Utilities for cost functions
'''

import numpy as np

EPSILON = 1e-10

def sigmoid_cross_entropy_cost(AL, Y):
    m = Y.shape[1]

    # Compute loss from AL and y.
    cost = (-1.0/m) * np.nansum(np.multiply(Y, np.log(AL + EPSILON)) + np.multiply((1-Y), np.log(1-AL + EPSILON)))

    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())

    return cost

def sigmoid_cross_entropy_dAL(AL, Y):
    return - (np.divide(Y, AL + EPSILON) - np.divide(1 - Y, 1 - AL + EPSILON))

def softmax_cross_entropy_cost(AL, Y):
    m = Y.shape[1]

    agreement = np.multiply(Y, np.log(AL + EPSILON))
    cost = (-1.0/m) * np.nansum(agreement)

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost

def softmax_cross_entropy_dZ(AL, Y):
    """
    NOTE: this cost derivative is w.r.t. Z of the last layer, rather than w.r.t. A
    That's because computing the softmax cross-entropy derivative w.r.t. A and
    the derivative of softmax w.r.t. Z separately and then composing them is much more costly.
    """
    return AL - Y