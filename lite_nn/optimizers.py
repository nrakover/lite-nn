"""
Interface and implementations of gradient optimizers to be used by simple_nn_lib
"""

class GradientOptimizer:
    def initialize(self, L, layer_dims):
        pass
    
    def gradient_step(self, L, parameters, grads, learning_rate):
        pass

import numpy as np

class VanillaOptimizer(GradientOptimizer):
    def gradient_step(self, L, parameters, grads, learning_rate):
        new_parameters = {}
        for l in range(1, L):
            new_parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * grads['dW' + str(l)]
            new_parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * grads['db' + str(l)]
        return new_parameters

class Adam(GradientOptimizer):
    def __init__(self, beta1=0.9, beta2=0.999,  epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 1
        self.V = {}
        self.S = {}
    
    def initialize(self, L, layer_dims):
        self.t = 1
        self.V.clear()
        self.S.clear()

        for l in range(1, L):
            self.V['dW' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
            self.V['db' + str(l)] = np.zeros((layer_dims[l], 1))

            self.S['dW' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
            self.S['db' + str(l)] = np.zeros((layer_dims[l], 1))
    
    def gradient_step(self, L, parameters, grads, learning_rate):
        V_corrected = {}
        S_corrected = {}
        new_parameters = {}

        for l in range(1, L):
            # update momentum terms V
            self.V['dW' + str(l)] = self.beta1 * self.V['dW' + str(l)] + (1-self.beta1) * grads['dW' + str(l)]
            self.V['db' + str(l)] = self.beta1 * self.V['db' + str(l)] + (1-self.beta1) * grads['db' + str(l)]

            # compute bias-corrected momentum terms
            V_corrected['dW' + str(l)] = self.V['dW' + str(l)] / (1 - self.beta1**self.t)
            V_corrected['db' + str(l)] = self.V['db' + str(l)] / (1 - self.beta1**self.t)

            # update RMSprop terms S
            self.S['dW' + str(l)] = self.beta2 * self.S['dW' + str(l)] + (1-self.beta2) * np.square(grads['dW' + str(l)])
            self.S['db' + str(l)] = self.beta2 * self.S['db' + str(l)] + (1-self.beta2) * np.square(grads['db' + str(l)])

            # compute bias-corrected RMSprop terms
            S_corrected['dW' + str(l)] = self.S['dW' + str(l)] / (1 - self.beta2**self.t)
            S_corrected['db' + str(l)] = self.S['db' + str(l)] / (1 - self.beta2**self.t)

            # compute new parameters
            new_parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * V_corrected['dW' + str(l)] / (np.sqrt(S_corrected['dW' + str(l)]) + self.epsilon)
            new_parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * V_corrected['db' + str(l)] / (np.sqrt(S_corrected['db' + str(l)]) + self.epsilon)
        
        # increment the internal update counter
        self.t += 1

        return new_parameters
