import numpy as np

def compute_L2_reg_cost(l2_lambda, params, L, m):
    l2_cost = 0.
    for l in range(1,L):
        W = params.get('W' + str(l))
        if W is not None:
            l2_cost += np.sum(np.square(W))
    return (l2_lambda / (2.0 * m)) * l2_cost
