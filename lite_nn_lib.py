'''
Simple NN library
'''

import numpy as np

from activation_layers import SigmoidLayer, ReluLayer
from cost_functions import SigmoidCrossEntropy
from initilizers import HeInit
import utils.regularization as reg_utils

class NN:
    '''
    Implements fully-connected neural networks
    '''

    def __init__(self, layer_dims, activation_fns=None, cost_fn=None, initializer=HeInit(), l2_lambda=0.0):
        self.L = len(layer_dims)
        self.layer_dims = layer_dims
        self.activation_fns = activation_fns if activation_fns is not None else [ReluLayer() for l in range(self.L - 2)] + [SigmoidLayer()] 
        self.cost_fn = cost_fn if cost_fn is not None else SigmoidCrossEntropy()
        self.initializer = initializer
        self.l2_lambda = l2_lambda
        self.parameters = {}

        assert (len(self.activation_fns) == len(self.layer_dims) - 1)
    
    def initialize_params(self):
        '''
        Initialize model parameters
        '''
        self.parameters = self.initializer.get_initial_params(self.L, self.layer_dims)
    
    @staticmethod
    def linear_forward(A_prev, W, b):
        '''
        Compute Z and the linear_cache
        '''
        Z = np.dot(W, A_prev) + b

        assert(Z.shape == (W.shape[0], A_prev.shape[1]))

        cache = (A_prev, W, b)

        return Z, cache

    def forward_propagatation(self, X):
        '''
        Computes the output of a full forward propagation

        returns
            AL: output of final layer
            caches: list of tuples of format (linear_cache, activation_cache), one per layer
        '''
        caches = []

        A = X
        for l in range(1, self.L):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]

            Z, linear_cache = NN.linear_forward(A_prev, W, b)
            A, activation_cache = self.activation_fns[l-1].forward(Z)

            assert (A.shape == (W.shape[0], A_prev.shape[1]))

            caches.append((linear_cache, activation_cache))
        
        return A, caches
    
    def compute_cost(self, AL, Y):
        '''
        Computes the cost of the final activation given the configured cost function
        '''
        return self.cost_fn.compute_cost(AL, Y) + reg_utils.compute_L2_reg_cost(self.l2_lambda, self.parameters, self.L, AL.shape[1])
    
    def compute_cost_derivative(self, AL, Y):
        '''
        Computes the derivative of the final activation relative to the cost given the configured cost function
        '''
        return self.cost_fn.compute_cost_derivative(AL, Y)
    
    @staticmethod
    def linear_backward(dZ, cache, l2_lambda):
        '''
        Compute dA_prev, dW, and db given dZ and the linear_cache
        '''
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1.0/m) * np.dot(dZ, A_prev.T) + (l2_lambda / m) * W
        db = (1.0/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db
    
    def back_propagation(self, AL, Y, caches):
        '''
        Computes the gradients of all parameters
        '''
        grads = {}
        Y = Y.reshape(AL.shape)

        dA = self.compute_cost_derivative(AL, Y)

        for l in reversed(range(1, self.L)):
            linear_cache, activation_cache = caches[l-1]
            
            dZ = self.activation_fns[l-1].backward(dA, activation_cache)
            dA_prev, dW, db = NN.linear_backward(dZ, linear_cache, self.l2_lambda)

            grads['dA' + str(l-1)] = dA_prev
            grads['dW' + str(l)] = dW
            grads['db' + str(l)] = db

            dA = dA_prev
        
        return grads
    
    def update_parameters(self, grads, learning_rate):
        '''
        Updates model parameters given the gradients and learning rate
        '''
        for l in range(1, self.L):
            self.parameters['W' + str(l)] = self.parameters['W' + str(l)] - learning_rate * grads['dW' + str(l)]
            self.parameters['b' + str(l)] = self.parameters['b' + str(l)] - learning_rate * grads['db' + str(l)]
    
    def fit(self, X, Y, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
        costs = []

        # initialize params
        self.initialize_params()

        # perform num_iterations of batch gradient descent
        for i in range(num_iterations):

            # forward propagation
            AL, caches = self.forward_propagatation(X)

            # compute cost
            cost = self.compute_cost(AL, Y)

            # back propagation
            grads = self.back_propagation(AL, Y, caches)

            # update params
            self.update_parameters(grads, learning_rate)

            costs.append(cost)
            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
        
        return self.parameters.copy(), costs
    
    def predict(self, X, threshold=0.5):
        AL, _ = self.forward_propagatation(X)
        predictions = (AL > threshold) * 1

        return predictions
    
    def score(self, X, Y, threshold=0.5):
        predictions = self.predict(X, threshold)
        return np.sum(predictions == Y) / predictions.size
