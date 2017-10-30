'''
Simple NN library
'''

import numpy as np

from activation_layers import SigmoidLayer, ReluLayer
from cost_functions import SigmoidCrossEntropy
from initilizers import HeInit
from optimizers import VanillaOptimizer
import utils.regularization as reg_utils
import utils.gradient_descent as gd

class NN:
    '''
    Implements fully-connected neural networks
    '''

    def __init__(self, layer_dims, activation_fns=None, cost_fn=None, initializer=HeInit(), optimizer=VanillaOptimizer(), l2_lambda=0.0, dropout_probs=None):
        self.L = len(layer_dims)
        self.layer_dims = layer_dims
        self.activation_fns = activation_fns if activation_fns is not None else [ReluLayer() for l in range(self.L - 2)] + [SigmoidLayer()] 
        self.cost_fn = cost_fn if cost_fn is not None else SigmoidCrossEntropy()
        self.initializer = initializer
        self.optimizer = optimizer
        self.l2_lambda = l2_lambda
        self.dropout_probs = dropout_probs + [0] if dropout_probs is not None else None
        self.parameters = {}

        assert (len(self.activation_fns) == len(self.layer_dims) - 1)
        assert (dropout_probs is None or len(self.dropout_probs) == len(self.activation_fns))
    
    def fit(self, X, Y, learning_rate = 0.0075, num_iterations = 3000, mini_batche_size=None, random_seed=None, print_cost=False):
        costs = []
        use_dropout = self.dropout_probs is not None
        mini_batche_size = mini_batche_size or X.shape[1]
        mini_batch_seed = random_seed

        # initialize params
        self.initialize_params()

        # initialize optimizer
        self.optimizer.initialize(self.L, self.layer_dims)

        # perform num_iterations epochs of gradient descent
        i = 0 # gradient descent counter
        for epoch in range(num_iterations):

            # create mini-batches
            mini_batch_seed = None if mini_batch_seed is None else mini_batch_seed + 1
            mini_batches = gd.random_mini_batches(X, Y, mini_batch_size=mini_batche_size, seed=mini_batch_seed)

            # perform mini-batch gradient descent on each mini-batch
            for (mini_X, mini_Y) in mini_batches:

                # forward propagation
                AL, caches = self.forward_propagatation(mini_X, dropout=use_dropout)

                # compute cost
                cost = self.compute_cost(AL, mini_Y)

                # back propagation
                grads = self.back_propagation(AL, mini_Y, caches, dropout=use_dropout)

                # update params
                self.update_parameters(grads, learning_rate)

                costs.append(cost)
                # Print the cost every 100 training example
                if print_cost and i % 100 == 0:
                    print ("Cost after iteration %i: %f" %(i, cost))
                i+=1
        
        return self.parameters.copy(), costs
    
    def predict(self, X, threshold=0.5):
        AL, _ = self.forward_propagatation(X)
        predictions = (AL > threshold) * 1

        return predictions
    
    def score(self, X, Y, threshold=0.5):
        predictions = self.predict(X, threshold)
        return np.average(predictions == Y)

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

    def forward_propagatation(self, X, dropout=False):
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
            
            # apply dropout
            if dropout:
                keep_prob = 1 - self.dropout_probs[l-1]
                A, D = reg_utils.forward_dropout(A, keep_prob)
                caches.append((linear_cache, activation_cache, D))
            else:
                caches.append((linear_cache, activation_cache))

            assert (A.shape == (W.shape[0], A_prev.shape[1]))
        
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
    
    def back_propagation(self, AL, Y, caches, dropout=False):
        '''
        Computes the gradients of all parameters
        '''
        grads = {}
        Y = Y.reshape(AL.shape)

        dA = self.compute_cost_derivative(AL, Y)

        for l in reversed(range(1, self.L)):
            if dropout:
                linear_cache, activation_cache, D = caches[l-1]
                # apply dropout
                keep_prob = 1 - self.dropout_probs[l-1]
                dA = reg_utils.backward_dropout(dA, keep_prob, D)
            else:
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
        self.parameters = self.optimizer.gradient_step(self.L, self.parameters, grads, learning_rate)
