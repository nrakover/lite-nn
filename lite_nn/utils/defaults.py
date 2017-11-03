"""
Some default configurations for different network structures
"""

from ..activation_layers import ReluLayer, SigmoidLayer, SoftmaxOutputLayer
from ..cost_functions import SigmoidCrossEntropy, SoftmaxCrossEntropy

def default_activations(layer_dims):
    return [ReluLayer() for l in range(len(layer_dims) - 2)] + [default_final_activation(layer_dims[-1])]

def default_final_activation(num_units_in_final_layer):
    if num_units_in_final_layer == 1:
        return SigmoidLayer()
    else:
        return SoftmaxOutputLayer()

def default_cost_function(num_units_in_final_layer):
    if num_units_in_final_layer == 1:
        return SigmoidCrossEntropy()
    else:
        return SoftmaxCrossEntropy()
