"""
Interface and implementations of weight initializers to be used by simple_nn_lib
"""

class WeightInitializer:
    def get_initial_params(self, L, layer_dims):
        '''
        Creates a dictionary of the W and b weights based off some initialization strategy

        Args:
            L (int): number of network layers (including input layer)
            layer_dims (list): number of nodes per layer
        
        Returns:
            a dictionary, keyed by the index-suffixed parameter names, of the
        initialized parameters (each represented as an np.array)
        '''
        pass

#######################################
#######     Implementations     #######
#######################################

import numpy as np

class SimpleRandomInit(WeightInitializer):
    def __init__(self, scaling_factor=0.01):
        self.scaling_factor = scaling_factor

    def get_initial_params(self, L, layer_dims):
        params = {}
        for l in range(1, L):
            params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * self.scaling_factor
            params['b' + str(l)] = np.zeros((layer_dims[l], 1))
        return params

class XavierInit(WeightInitializer):
    def get_initial_params(self, L, layer_dims):
        params = {}
        for l in range(1, L):
            params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1.0/layer_dims[l-1])
            params['b' + str(l)] = np.zeros((layer_dims[l], 1))
        return params

class HeInit(WeightInitializer):
    def get_initial_params(self, L, layer_dims):
        params = {}
        for l in range(1, L):
            params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0/layer_dims[l-1])
            params['b' + str(l)] = np.zeros((layer_dims[l], 1))
        return params
