import numpy as np

def random_mini_batches(X, Y, mini_batch_size=64, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    m = X.shape[1]
    permutation = list(np.random.permutation(m))

    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    mini_batches = []
    for t in range(int(m/mini_batch_size)):
        X_t = shuffled_X[:, t*mini_batch_size : (t+1)*mini_batch_size]
        Y_t = shuffled_Y[:, t*mini_batch_size : (t+1)*mini_batch_size]
        mini_batches.append((X_t, Y_t))
    
    if m % mini_batch_size != 0:
        X_last = shuffled_X[:, -(m % mini_batch_size):]
        Y_last = shuffled_Y[:, -(m % mini_batch_size):]
        mini_batches.append((X_last, Y_last))
    
    return mini_batches
