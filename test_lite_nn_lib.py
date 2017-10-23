"""
Basic unit tests for simple_nn_lib.py
"""

import unittest
import numpy as np

import lite_nn_lib as nn
import test_utils.coursera_test_cases as test_cases

class TestNN(unittest.TestCase):
    
    def assertDeepEquals(self, A, B):
        self.assertTupleEqual(A.shape, B.shape)
        delta = np.abs(A - B)
        self.assertTrue((delta < 0.00000001).all())
    
    def test_compute_cost(self):
        Y, AL, expected = test_cases.compute_cost_test_case()
        model = nn.NN([1, Y.shape[0]])
        cost = model.compute_cost(AL, Y)
        self.assertAlmostEqual(expected, cost, places=16)
    
    def test_linear_forward(self):
        A, W, b, expected = test_cases.linear_forward_test_case()
        Z, _ = nn.NN.linear_forward(A, W, b)
        self.assertDeepEquals(expected, Z)
    
    def test_linear_backward(self):
        dZ, linear_cache, expected_dA_prev, expected_dW, expected_db = test_cases.linear_backward_test_case()
        dA_prev, dW, db = nn.NN.linear_backward(dZ, linear_cache)
        self.assertDeepEquals(expected_dA_prev, dA_prev)
        self.assertDeepEquals(expected_dW, dW)
        self.assertDeepEquals(expected_db, db)
    
    def test_update_parameters(self):
        layer_dims, parameters, grads, expected_W1, expected_b1, expected_W2, expected_b2 = test_cases.update_parameters_test_case()
        model = nn.NN(layer_dims)
        model.parameters = parameters

        model.update_parameters(grads, 0.1)
        self.assertDeepEquals(model.parameters['W1'], expected_W1)
        self.assertDeepEquals(model.parameters['b1'], expected_b1)
        self.assertDeepEquals(model.parameters['W2'], expected_W2)
        self.assertDeepEquals(model.parameters['b2'], expected_b2)
    
    def test_forward_propagation(self):
        X, parameters, layer_dims, expected_AL = test_cases.L_model_forward_test_case_2hidden()
        model = nn.NN(layer_dims)
        model.parameters = parameters

        AL, caches = model.forward_propagatation(X)

        self.assertDeepEquals(AL, expected_AL)
        self.assertEqual(len(caches), 3)
    
    def test_back_prop(self):
        AL, Y_assess, caches, layer_dims, expected_dW1, expected_db1, expected_dA1 = test_cases.L_model_backward_test_case()
        model = nn.NN(layer_dims)

        grads = model.back_propagation(AL, Y_assess, caches)

        self.assertDeepEquals(grads['dW1'], expected_dW1)
        self.assertDeepEquals(grads['db1'], expected_db1)
        self.assertDeepEquals(grads['dA1'], expected_dA1)

if __name__ == '__main__':
    unittest.main()
