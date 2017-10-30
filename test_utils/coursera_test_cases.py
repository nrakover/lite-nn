import numpy as np

def linear_forward_test_case():
    np.random.seed(1)
    """
    X = np.array([[-1.02387576, 1.12397796],
 [-1.62328545, 0.64667545],
 [-1.74314104, -0.59664964]])
    W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
    b = np.array([[1]])
    """
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)

    expected = np.array([[ 3.26295337, -1.23429987]])
    
    return A, W, b, expected

def linear_activation_forward_test_case():
    """
    X = np.array([[-1.02387576, 1.12397796],
 [-1.62328545, 0.64667545],
 [-1.74314104, -0.59664964]])
    W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
    b = 5
    """
    np.random.seed(2)
    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    return A_prev, W, b

def L_model_forward_test_case():
    """
    X = np.array([[-1.02387576, 1.12397796],
 [-1.62328545, 0.64667545],
 [-1.74314104, -0.59664964]])
    parameters = {'W1': np.array([[ 1.62434536, -0.61175641, -0.52817175],
        [-1.07296862,  0.86540763, -2.3015387 ]]),
 'W2': np.array([[ 1.74481176, -0.7612069 ]]),
 'b1': np.array([[ 0.],
        [ 0.]]),
 'b2': np.array([[ 0.]])}
    """
    np.random.seed(1)
    X = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return X, parameters

def compute_cost_test_case():
    Y = np.asarray([[1, 1, 1]])
    aL = np.array([[.8,.9,0.4]])
    expected = 0.41493159961539694
    
    return Y, aL, expected

def linear_backward_test_case():
    """
    z, linear_cache = (np.array([[-0.8019545 ,  3.85763489]]), (np.array([[-1.02387576,  1.12397796],
       [-1.62328545,  0.64667545],
       [-1.74314104, -0.59664964]]), np.array([[ 0.74505627,  1.97611078, -1.24412333]]), np.array([[1]]))
    """
    np.random.seed(1)
    dZ = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    linear_cache = (A, W, b)

    expected_dA_prev = np.array([[ 0.51822968, -0.19517421], [-0.40506361, 0.15255393], [ 2.37496825, -0.89445391]])
    expected_dW = np.array([[-0.10076895, 1.40685096, 1.64992505]])
    expected_db = np.array([[ 0.50629448]])
    return dZ, linear_cache, expected_dA_prev, expected_dW, expected_db

def linear_activation_backward_test_case():
    """
    aL, linear_activation_cache = (np.array([[ 3.1980455 ,  7.85763489]]), ((np.array([[-1.02387576,  1.12397796], [-1.62328545,  0.64667545], [-1.74314104, -0.59664964]]), np.array([[ 0.74505627,  1.97611078, -1.24412333]]), 5), np.array([[ 3.1980455 ,  7.85763489]])))
    """
    np.random.seed(2)
    dA = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    Z = np.random.randn(1,2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)
    
    return dA, linear_activation_cache

def L_model_backward_test_case():
    """
    X = np.random.rand(3,2)
    Y = np.array([[1, 1]])
    parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747]]), 'b1': np.array([[ 0.]])}

    aL, caches = (np.array([[ 0.60298372,  0.87182628]]), [((np.array([[ 0.20445225,  0.87811744],
           [ 0.02738759,  0.67046751],
           [ 0.4173048 ,  0.55868983]]),
    np.array([[ 1.78862847,  0.43650985,  0.09649747]]),
    np.array([[ 0.]])),
   np.array([[ 0.41791293,  1.91720367]]))])
   """
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    Z1 = np.random.randn(3,2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3,2)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    Z2 = np.random.randn(1,2)
    linear_cache_activation_2 = ((A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)

    layer_dims = [4, 3, 1]

    expected_dW1 = np.array([[ 0.41010002, 0.07807203, 0.13798444, 0.10502167], [ 0., 0., 0., 0., ], [ 0.05283652, 0.01005865, 0.01777766, 0.0135308 ]])
    expected_db1 = np.array([[-0.22007063], [ 0. ], [-0.02835349]])
    expected_dA1 = np.array([[ 0.12913162, -0.44014127], [-0.14175655, 0.48317296], [ 0.01663708, -0.05670698]])

    return AL, Y, caches, layer_dims, expected_dW1, expected_db1, expected_dA1

def update_parameters_test_case():
    """
    parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747],
        [-1.8634927 , -0.2773882 , -0.35475898],
        [-0.08274148, -0.62700068, -0.04381817],
        [-0.47721803, -1.31386475,  0.88462238]]),
 'W2': np.array([[ 0.88131804,  1.70957306,  0.05003364, -0.40467741],
        [-0.54535995, -1.54647732,  0.98236743, -1.10106763],
        [-1.18504653, -0.2056499 ,  1.48614836,  0.23671627]]),
 'W3': np.array([[-1.02378514, -0.7129932 ,  0.62524497],
        [-0.16051336, -0.76883635, -0.23003072]]),
 'b1': np.array([[ 0.],
        [ 0.],
        [ 0.],
        [ 0.]]),
 'b2': np.array([[ 0.],
        [ 0.],
        [ 0.]]),
 'b3': np.array([[ 0.],
        [ 0.]])}
    grads = {'dW1': np.array([[ 0.63070583,  0.66482653,  0.18308507],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ]]),
 'dW2': np.array([[ 1.62934255,  0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ]]),
 'dW3': np.array([[-1.40260776,  0.        ,  0.        ]]),
 'da1': np.array([[ 0.70760786,  0.65063504],
        [ 0.17268975,  0.15878569],
        [ 0.03817582,  0.03510211]]),
 'da2': np.array([[ 0.39561478,  0.36376198],
        [ 0.7674101 ,  0.70562233],
        [ 0.0224596 ,  0.02065127],
        [-0.18165561, -0.16702967]]),
 'da3': np.array([[ 0.44888991,  0.41274769],
        [ 0.31261975,  0.28744927],
        [-0.27414557, -0.25207283]]),
 'db1': 0.75937676204411464,
 'db2': 0.86163759922811056,
 'db3': -0.84161956022334572}
    """
    np.random.seed(2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3,4)
    db1 = np.random.randn(3,1)
    dW2 = np.random.randn(1,3)
    db2 = np.random.randn(1,1)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    layer_dims = [4,3,1]

    expected_W1 = np.array([[-0.59562069, -0.09991781, -2.14584584, 1.82662008], [-1.76569676, -0.80627147, 0.51115557, -1.18258802], [-1.0535704, -0.86128581, 0.68284052, 2.20374577]])
    expected_b1 = np.array([[-0.04659241], [-1.28888275], [ 0.53405496]])
    expected_W2 = np.array([[-0.55569196, 0.0354055, 1.32964895]])
    expected_b2 = np.array([[-0.84610769]])
    
    return layer_dims, parameters, grads, expected_W1, expected_b1, expected_W2, expected_b2


def L_model_forward_test_case_2hidden():
    np.random.seed(6)
    X = np.random.randn(5,4)
    W1 = np.random.randn(4,5)
    b1 = np.random.randn(4,1)
    W2 = np.random.randn(3,4)
    b2 = np.random.randn(3,1)
    W3 = np.random.randn(1,3)
    b3 = np.random.randn(1,1)
  
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    layer_dims = [5, 4, 3, 1]

    expected_AL = np.array([[ 0.03921668, 0.70498921, 0.19734387, 0.04728177]])
    
    return X, parameters, layer_dims, expected_AL

def print_grads(grads):
    print ("dW1 = "+ str(grads["dW1"]))
    print ("db1 = "+ str(grads["db1"]))
    print ("dA1 = "+ str(grads["dA2"])) # this is done on purpose to be consistent with lecture where we normally start with A0
                                        # in this implementation we started with A1, hence we bump it up by 1. 
    
def compute_cost_with_regularization_test_case():
    np.random.seed(1)
    Y_assess = np.array([[1, 1, 0, 1, 0]])
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    a3 = np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]])

    layer_dims = [3, 2, 3, 1]
    l2_lambda = 0.1
    expected_cost = 1.78648594516
    return a3, Y_assess, parameters, layer_dims, l2_lambda, expected_cost

def backward_propagation_with_regularization_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(3, 5)
    Y_assess = np.array([[1, 1, 0, 1, 0]])
    Z1 = np.array([[-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],[-1.98043538,  4.1600994 ,  0.79051021,  1.46493512, -0.45506242]])
    A1 = np.array([[ 0.        ,  3.32524635,  2.13994541,  2.60700654,  0.        ],[ 0.        ,  4.1600994 ,  0.79051021,  1.46493512,  0.        ]])
    W1 = np.array([[-1.09989127, -0.17242821, -0.87785842],[ 0.04221375,  0.58281521, -1.10061918]])
    b1 = np.array([[ 1.14472371],[ 0.90159072]])
    Z2 = np.array([[ 0.53035547,  5.94892323,  2.31780174,  3.16005701,  0.53035547],[-0.69166075, -3.47645987, -2.25194702, -2.65416996, -0.69166075],[-0.39675353, -4.62285846, -2.61101729, -3.22874921, -0.39675353]])
    A2 = np.array([[ 0.53035547,  5.94892323,  2.31780174,  3.16005701,  0.53035547],[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])
    W2 = np.array([[ 0.50249434,  0.90085595],[-0.68372786, -0.12289023],[-0.93576943, -0.26788808]])
    b2 = np.array([[ 0.53035547],[-0.69166075],[-0.39675353]])
    Z3 = np.array([[-0.3771104 , -4.10060224, -1.60539468, -2.18416951, -0.3771104 ]])
    A3 = np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]])
    W3 = np.array([[-0.6871727 , -0.84520564, -0.67124613]])
    b3 = np.array([[-0.0126646]])

    caches = [
        ((X_assess, W1, b1), Z1),
        ((A1, W2, b2), Z2),
        ((A2, W3, b3), Z3)
    ]

    layer_dims = [3, 2, 3, 1]
    l2_lambda = 0.7
    expected_dW1 = np.array([[-0.25604647,  0.1229883,  -0.28297132], [-0.17706304,  0.345361,  -0.44105717]])
    expected_dW2 = np.array([[ 0.79276497,  0.85133932], [-0.0957219,  -0.01720463], [-0.13100772, -0.03750433]])
    expected_dW3 = np.array([[-1.77691375, -0.11832879, -0.09397446]])
    return A3, Y_assess, caches, layer_dims, l2_lambda, expected_dW1, expected_dW2, expected_dW3

def forward_propagation_with_dropout_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(3, 5)
    W1 = np.random.randn(2, 3)
    b1 = np.random.randn(2, 1)
    W2 = np.random.randn(3, 2)
    b2 = np.random.randn(3, 1)
    W3 = np.random.randn(1, 3)
    b3 = np.random.randn(1, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    layer_dims = [3, 2, 3, 1]
    dropout_probs = [0.3, 0.3]
    expected_AL = np.array([[ 0.36974721,  0.00305176,  0.04565099,  0.49683389,  0.36974721]])
    np.random.seed(1)
    return X_assess, parameters, layer_dims, dropout_probs, expected_AL

def backward_propagation_with_dropout_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(3, 5)
    Y_assess = np.array([[1, 1, 0, 1, 0]])
    Z1 = np.array([[-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],[-1.98043538,  4.1600994 ,  0.79051021,  1.46493512, -0.45506242]])
    D1 = np.array([[ True, False,  True,  True,  True],[ True,  True,  True,  True, False]], dtype=bool)
    A1 = np.array([[ 0.        ,  0.        ,  4.27989081,  5.21401307,  0.        ],[ 0.        ,  8.32019881,  1.58102041,  2.92987024,  0.        ]])
    W1 = np.array([[-1.09989127, -0.17242821, -0.87785842],[ 0.04221375,  0.58281521, -1.10061918]])
    b1 = np.array([[ 1.14472371],[ 0.90159072]])
    Z2 = np.array([[ 0.53035547,  8.02565606,  4.10524802,  5.78975856,  0.53035547],[-0.69166075, -1.71413186, -3.81223329, -4.61667916, -0.69166075],[-0.39675353, -2.62563561, -4.82528105, -6.0607449 , -0.39675353]])
    D2 = np.array([[ True, False,  True, False,  True],[False,  True, False,  True,  True],[False, False,  True, False, False]], dtype=bool)
    A2 = np.array([[ 1.06071093,  0.        ,  8.21049603,  0.        ,  1.06071093],[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])
    W2 = np.array([[ 0.50249434,  0.90085595],[-0.68372786, -0.12289023],[-0.93576943, -0.26788808]])
    b2 = np.array([[ 0.53035547],[-0.69166075],[-0.39675353]])
    Z3 = np.array([[-0.7415562 , -0.0126646 , -5.65469333, -0.0126646 , -0.7415562 ]])
    A3 = np.array([[ 0.32266394,  0.49683389,  0.00348883,  0.49683389,  0.32266394]])
    W3 = np.array([[-0.6871727 , -0.84520564, -0.67124613]])
    b3 = np.array([[-0.0126646]])
    
    caches = [
        ((X_assess, W1, b1), Z1, D1),
        ((A1, W2, b2), Z2, D2),
        ((A2, W3, b3), Z3, np.ones(Z3.shape, dtype=bool))
    ]

    layer_dims = [3, 2, 3, 1]
    dropout_probs = [0.2, 0.2]

    expected_dA1 = np.array([[ 0.36544439,  0.,         -0.00188233,  0.,         -0.17408748],[ 0.65515713,  0.,         -0.00337459,  0.,         -0.        ]])
    expected_dA2 = np.array([[ 0.58180856,  0.,         -0.00299679,  0.,         -0.27715731],[ 0.,          0.53159854, -0.,          0.53159854, -0.34089673],[ 0. ,         0. ,        -0.00292733,  0. ,        -0.        ]])

    return X_assess, Y_assess, A3, caches, layer_dims, dropout_probs, expected_dA1, expected_dA2

def update_parameters_with_adam_test_case():
    np.random.seed(1)
    v, s = ({'dW1': np.array([[ 0.,  0.,  0.],
         [ 0.,  0.,  0.]]), 'dW2': np.array([[ 0.,  0.,  0.],
         [ 0.,  0.,  0.],
         [ 0.,  0.,  0.]]), 'db1': np.array([[ 0.],
         [ 0.]]), 'db2': np.array([[ 0.],
         [ 0.],
         [ 0.]])}, {'dW1': np.array([[ 0.,  0.,  0.],
         [ 0.,  0.,  0.]]), 'dW2': np.array([[ 0.,  0.,  0.],
         [ 0.,  0.,  0.],
         [ 0.,  0.,  0.]]), 'db1': np.array([[ 0.],
         [ 0.]]), 'db2': np.array([[ 0.],
         [ 0.],
         [ 0.]])})
    W1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    W2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)

    dW1 = np.random.randn(2,3)
    db1 = np.random.randn(2,1)
    dW2 = np.random.randn(3,3)
    db2 = np.random.randn(3,1)
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    t = 2
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    learning_rate = 0.01
    L = 3

    expected_W1 = np.array([[ 1.63178673, -0.61919778, -0.53561312],[-1.08040999,  0.85796626, -2.29409733]])
    expected_b1 = np.array([[ 1.75225313], [-0.75376553]])
    expected_W2 = np.array([[ 0.32648046, -0.25681174, 1.46954931],[-2.05269934, -0.31497584, -0.37661299],[ 1.14121081, -1.09244991, -0.16498684]])
    expected_b2 = np.array([[-0.88529979],[ 0.03477238],[ 0.57537385]])

    expected_V_dW1 = np.array([[-0.11006192,  0.11447237,  0.09015907],[ 0.05024943,  0.09008559, -0.06837279]])
    expected_V_db1 = np.array([[-0.01228902],[-0.09357694]])
    expected_V_dW2 = np.array([[-0.02678881,  0.05303555, -0.06916608],[-0.03967535, -0.06871727, -0.08452056],[-0.06712461, -0.00126646, -0.11173103]])
    expected_V_db2 = np.array([[ 0.02344157],[ 0.16598022],[ 0.07420442]])

    expected_S_dW1 = np.array([[ 0.00121136,  0.00131039,  0.00081287],[ 0.0002525,   0.00081154,  0.00046748]])
    expected_S_db1 = np.array([[  1.51020075e-05],[  8.75664434e-04]])
    expected_S_dW2 = np.array([[  7.17640232e-05,   2.81276921e-04,   4.78394595e-04],[  1.57413361e-04,   4.72206320e-04,   7.14372576e-04],[  4.50571368e-04,   1.60392066e-07,   1.24838242e-03]])
    expected_S_db2 = np.array([[  5.49507194e-05],[  2.75494327e-03],[  5.50629536e-04]])
    
    return parameters, grads, learning_rate, L, v, s, t, beta1, beta2, epsilon, expected_W1, expected_b1, expected_W2, expected_b2, expected_V_dW1, expected_V_db1, expected_V_dW2, expected_V_db2, expected_S_dW1, expected_S_db1, expected_S_dW2, expected_S_db2
