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
    @staticmethod
    def backward(dA, activation_cache):
        """
        Args:
            dA (np.array): derivative of cost function with respect to the activation
            activation_cache (tuple): cache created during forward pass

        Returns:
            dZ (np.array): derivative of cost function with respect to the linear output
        """
        pass

#######################################
#######     Implementations     #######
#######################################

from .utils import activation as act

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

class SoftmaxOutputLayer(ActivationLayer):
    """
    NOTE: this activation layer can only be used as the *final* layer
    """
    @staticmethod
    def forward(Z):
        return act.softmax(Z)
    
    @staticmethod
    def backward(dZ, activation_cache):
        """
        NOTE: this is a bit of a hack, since we're actually passing dZ masquerading as dA
        This assumes that the softmax layer is used exclusively as the final layer, in
        conjunction with the softmax cross-entropy cost function
        """
        return dZ
