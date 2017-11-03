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
