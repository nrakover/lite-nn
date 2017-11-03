"""
Interface and implementations of cost functions to be used by simple_nn_lib
"""

class CostFunction:
    @staticmethod
    def compute_cost(AL, Y):
        """
        Computes cost based on final activation layer and expected outputs
        """
        pass

    @staticmethod
    def compute_cost_derivative(AL, Y):
        """
        Computes derivative of the cost function with respect to the final activation
        """
        pass

#######################################
#######     Implementations     #######
#######################################

from .utils import cost as cost

class SigmoidCrossEntropy(CostFunction):
    @staticmethod
    def compute_cost(AL, Y):
        return cost.sigmoid_cross_entropy_cost(AL, Y)
    @staticmethod
    def compute_cost_derivative(AL, Y):
        return cost.sigmoid_cross_entropy_dAL(AL, Y)

class SoftmaxCrossEntropy(CostFunction):
    """
    NOTE: this cost function can only be used with the softmax activation in the final layer
    """
    @staticmethod
    def compute_cost(AL, Y):
        return cost.softmax_cross_entropy_cost(AL, Y)
    @staticmethod
    def compute_cost_derivative(AL, Y):
        """
        NOTE: this is a bit of a hack, where we actually compute the cost with respect to Z directly
        """
        return cost.softmax_cross_entropy_dZ(AL, Y)
