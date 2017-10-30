"""
Basic unit tests for simple_nn_lib.py
"""

import unittest
import numpy as np

import lite_nn_lib as nn
import optimizers
import test_utils.coursera_test_cases as test_cases

class TestNN(unittest.TestCase):
    
    def assertNpArrayEquals(self, A, B, msg=None):
        shape_msg = None if msg is None else msg + " -- shape mismatch"
        self.assertTupleEqual(A.shape, B.shape, msg=shape_msg)
        delta = np.abs(A - B)
        self.assertTrue((delta < 0.00000001).all(), msg=msg)
    
    def test_compute_cost(self):
        Y, AL, expected = test_cases.compute_cost_test_case()
        model = nn.NN([1, Y.shape[0]])
        cost = model.compute_cost(AL, Y)
        self.assertAlmostEqual(expected, cost)
    
    def test_linear_forward(self):
        A, W, b, expected = test_cases.linear_forward_test_case()
        Z, _ = nn.NN.linear_forward(A, W, b)
        self.assertNpArrayEquals(expected, Z)
    
    def test_linear_backward(self):
        dZ, linear_cache, expected_dA_prev, expected_dW, expected_db = test_cases.linear_backward_test_case()
        dA_prev, dW, db = nn.NN.linear_backward(dZ, linear_cache, 0)
        self.assertNpArrayEquals(expected_dA_prev, dA_prev)
        self.assertNpArrayEquals(expected_dW, dW)
        self.assertNpArrayEquals(expected_db, db)
    
    def test_update_parameters(self):
        layer_dims, parameters, grads, expected_W1, expected_b1, expected_W2, expected_b2 = test_cases.update_parameters_test_case()
        model = nn.NN(layer_dims)
        model.parameters = parameters

        model.update_parameters(grads, 0.1)
        self.assertNpArrayEquals(model.parameters['W1'], expected_W1)
        self.assertNpArrayEquals(model.parameters['b1'], expected_b1)
        self.assertNpArrayEquals(model.parameters['W2'], expected_W2)
        self.assertNpArrayEquals(model.parameters['b2'], expected_b2)
    
    def test_forward_propagation(self):
        X, parameters, layer_dims, expected_AL = test_cases.L_model_forward_test_case_2hidden()
        model = nn.NN(layer_dims)
        model.parameters = parameters

        AL, caches = model.forward_propagatation(X)

        self.assertNpArrayEquals(AL, expected_AL)
        self.assertEqual(len(caches), 3)
    
    def test_back_prop(self):
        AL, Y_assess, caches, layer_dims, expected_dW1, expected_db1, expected_dA1 = test_cases.L_model_backward_test_case()
        model = nn.NN(layer_dims)

        grads = model.back_propagation(AL, Y_assess, caches)

        self.assertNpArrayEquals(grads['dW1'], expected_dW1)
        self.assertNpArrayEquals(grads['db1'], expected_db1)
        self.assertNpArrayEquals(grads['dA1'], expected_dA1)
    
    def test_L2_regularization_cost(self):
        A3, Y_assess, parameters, layer_dims, l2_lambda, expected_cost = test_cases.compute_cost_with_regularization_test_case()

        model = nn.NN(layer_dims, l2_lambda=l2_lambda)
        model.parameters = parameters
        cost = model.compute_cost(A3, Y_assess)
        self.assertAlmostEqual(cost, expected_cost)
    
    def test_L2_regularization_back_prop(self):
        AL_assess, Y_assess, caches, layer_dims, l2_lambda, expected_dW1, expected_dW2, expected_dW3 = test_cases.backward_propagation_with_regularization_test_case()

        model = nn.NN(layer_dims, l2_lambda=l2_lambda)
        grads = model.back_propagation(AL_assess, Y_assess, caches)

        self.assertNpArrayEquals(grads['dW1'], expected_dW1, "dW1 mismatch")
        self.assertNpArrayEquals(grads['dW2'], expected_dW2, "dW2 mismatch")
        self.assertNpArrayEquals(grads['dW3'], expected_dW3, "dW3 mismatch")
    
    def test_forward_prop_with_dropout(self):
        X_assess, parameters, layer_dims, dropout_probs, expected_AL = test_cases.forward_propagation_with_dropout_test_case()

        model = nn.NN(layer_dims, dropout_probs=dropout_probs)
        model.parameters = parameters

        AL, _ = model.forward_propagatation(X_assess, dropout=True)
        self.assertNpArrayEquals(expected_AL, AL)
    
    def test_back_prop_with_dropout(self):
        X_assess, Y_assess, AL, caches, layer_dims, dropout_probs, expected_dA1, expected_dA2 = test_cases.backward_propagation_with_dropout_test_case()

        model = nn.NN(layer_dims, dropout_probs=dropout_probs)
        
        grads = model.back_propagation(AL, Y_assess, caches, dropout=True)
        self.assertNpArrayEquals(expected_dA1, grads['dA1'])
        self.assertNpArrayEquals(expected_dA2, grads['dA2'])
    
    def test_adam_parameter_update(self):
        parameters, grads, learning_rate, L, v, s, t, beta1, beta2, epsilon, expected_W1, expected_b1, expected_W2, expected_b2, expected_V_dW1, expected_V_db1, expected_V_dW2, expected_V_db2, expected_S_dW1, expected_S_db1, expected_S_dW2, expected_S_db2 = test_cases.update_parameters_with_adam_test_case()

        adam = optimizers.Adam(beta1, beta2, epsilon)
        adam.t = t
        adam.V = v
        adam.S = s

        new_parameters = adam.gradient_step(L, parameters, grads, learning_rate)
        self.assertNpArrayEquals(expected_W1, new_parameters['W1'])
        self.assertNpArrayEquals(expected_b1, new_parameters['b1'])
        self.assertNpArrayEquals(expected_W2, new_parameters['W2'])
        self.assertNpArrayEquals(expected_b2, new_parameters['b2'])

        self.assertNpArrayEquals(expected_V_dW1, adam.V['dW1'])
        self.assertNpArrayEquals(expected_V_db1, adam.V['db1'])
        self.assertNpArrayEquals(expected_V_dW2, adam.V['dW2'])
        self.assertNpArrayEquals(expected_V_db2, adam.V['db2'])

        self.assertNpArrayEquals(expected_S_dW1, adam.S['dW1'])
        self.assertNpArrayEquals(expected_S_db1, adam.S['db1'])
        self.assertNpArrayEquals(expected_S_dW2, adam.S['dW2'])
        self.assertNpArrayEquals(expected_S_db2, adam.S['db2'])

if __name__ == '__main__':
    unittest.main()
