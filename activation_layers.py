"""
Interface and implementations of activations layers to be used by simple_nn_lib
"""

class ActivationLayer:
    @staticmethod
    def forward(Z):
        """
        Args:
            Z (np.array): linear output

        Returns:
            A (np.array): activation output
            activation_cache (tuple): cache to be passed in to backwards pass
        """
        pass

#######################################
#######     Implementations     #######
#######################################

import utils.activation as act

class SigmoidLayer(ActivationLayer):
    @staticmethod
    def forward(Z):
        return act.sigmoid(Z)
    @staticmethod
    def backward(dA, activation_cache):
        return act.sigmoid_backward(dA, activation_cache)

class ReluLayer(ActivationLayer):
    @staticmethod
    def forward(Z):
        return act.relu(Z)
    @staticmethod
    def backward(dA, activation_cache):
        return act.relu_backward(dA, activation_cache)
