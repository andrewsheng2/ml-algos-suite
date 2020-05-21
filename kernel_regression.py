import numpy as np

def boxcar_kernel(x, z, width):
    """
    Return 1 if the Eudclidean distance between input vectors is
    less than or equal to width/2, and return zero otherwise
    """
    dist = np.linalg.norm(x-z)
    if dist <= width / 2: return 1
    else: return 0


def linear_kernel(x, z):
    """
    Return the result of taking the dot product of the two
    input vectors
    """
    return np.asscalar(x.T @ z)


def polynomial_kernel(x, z, d):
    """
    Return the result of applying the polynomial kernel of degree
    up to d on the two input vectors: K(x, z) = (x^Tz+1)^d
    """
    return np.asscalar(((x.T @ z) + 1)**d)


def rbf_kernel(x, z, gamma):
    """
    Return the result of applying the radial basis function kernel
    on the two input vectors, given the hyperparameter gamma
    """
    dist = np.linalg.norm(x-z)
    return np.asscalar(np.exp(-1*gamma*(dist)**2))


def kernel_regression(X_train, y_train, X_val, y_val, kernel, lmbd, hyp):
    """
    Use kernel regression with given kernel function to predict points

    Inputs:
    X_train: (N_train,M) numpy ndarray, N_train examples, M features
    y_train: (N_train,1) numpy ndarray, labels of training data
    X_val: (N_val, M) numpy ndarray, N_val examples, M features
    y_train: (N_val,1) numpy ndarray, labels of validation data
    kernel: string indicating which kernel function to use
            "boxcar", "rbf", "linear", "polynomial"
    lmbd: scalar lambda value for regularization
    hyp: hyperparameter for kernel functions that need it
         (width for boxcar, gamma for rbf, d for polynomial)

    Returns:
    predict_points: function to predict a new point
    mse_train: MSE for training data (lower is better)
    mse_val: MSE for validation data (lower is better)
    """
    def kernel_function(x, z):
        if kernel == "Boxcar": return boxcar_kernel(x, z, hyp)
        if kernel == "Linear": return linear_kernel(x, z)
        if kernel == "Polynomial": return polynomial_kernel(x, z, hyp)
        if kernel == "RBF": return boxcar_kernel(x, z, hyp)

    N_train, M = X_train.shape
    K = np.zeros([N_train,N_train])
    for i in range(N_train):
        for j in range(N_train):
            if i >= j:
                k = kernel_function(X_train[i], X_train[j])
                K[i][j] = k
                K[j][i] = k
    alpha = np.linalg.pinv(K + lmbd*np.identity(N_train)) @ y_train

    y_hat_train = np.zeros((N_train,1))
    for i in range(N_train):
        x_i = np.reshape(X_train[i], (M,1))
        for j in range(N_train):
            x_j = np.reshape(X_train[j], (M,1))
            y_hat_train[i] += alpha[j] * kernel_function(x_i, x_j)
    mse_train = mse(y_train, y_hat_train)

    def predict_points(X):
        N, M = X.shape
        y_hat = np.zeros((N,1))
        for i in range(N):
            x_i = np.reshape(X[i], (M,1))
            for j in range(N_train):
                x_j = np.reshape(X_train[j], (M,1))
                y_hat[i] += alpha[j] * kernel_function(x_i, x_j)
        return y_hat

    y_hat_val = predict_points(X_val)
    mse_val = mse(y_val, y_hat_val)

    return (predict_points, mse_train, mse_val)

def mse(y, y_hat):
    err = y - y_hat
    sqerr = err**2
    return np.mean(sqerr)
