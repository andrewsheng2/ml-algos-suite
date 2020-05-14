import numpy as np

def objective(X, y, w, lmbd):
    """
    Return the objective function for L2-regularized logistic regression

    Inputs:
    X: (N,M) numpy ndarray, N examples, M features
    y: (N,1) numpy ndarray, labels of training data

    Returns:
    objective: scalar value for the negative log likelihood
    """
    objective = 0
    N = X.shape[0]
    for i in range(N):
        x_i = X[i]
        y_i = y[i]
        mu_i = logistic(w.T @ x_i)
        objective += np.asscalar(y_i*np.log(mu_i) + (1-y_i)*np.log(1-mu_i))
    objective *= -1
    objective += np.asscalar(lmbd*(1/2)*(np.linalg.norm(w))**2)
    return objective

def logistic(z):
    return 1 / (1 + np.exp(-1*z))

def gradient(X, y, w, lmbd):
    N = X.shape[0]
    M = X.shape[1]
    grad = lmbd*w
    for i in range(N):
        y_i = np.asscalar(y[i])
        x_i = np.reshape(X[i], (M,1))
        mu_i = np.asscalar(logistic(w.T @ x_i))
        tmp = np.reshape((mu_i - y_i)*x_i, (M,))
        grad = grad + tmp
    return grad

def logistic_gradient(X_train, y_train, X_val, y_val, lmbd, alpha, num_iter):
    """
    Implement gradient descent on the objective function to find
    weights for L2-regularized logistic regression

    Inputs:
    X_train: (N_train,M) numpy ndarray, N_train examples, M features
    y_train: (N_train,1) numpy ndarray, labels of training data
    X_val: (N_val, M) numpy ndarray, N_val examples, M features
    y_train: (N_val,1) numpy ndarray, labels of validation data
    lmbd: scalar lambda value for regularization
    alpha: scalar learning rate for gradient descent
    num_iter: number of iterations to run gradient descent

    Returns:
    w: optimal weights for regression based on gradient descent
    obj_train: objective value for training data (lower is better)
    obj_val: objective value for validation data (lower is better)
    """
    N_train, M = X_train.shape
    N_val = X_val.shape[0]

    # add column of ones in front for bias term
    X_train_ones = np.ones([N_train,1])
    X_train = np.concatenate((X_train_ones, X_train), axis=1)
    X_val_ones = np.ones([N_val,1])
    X_val = np.concatenate((X_val_ones, X_val), axis=1)

    w = np.random.rand(M+1)
    for i in range(num_iter):
        grad = gradient(X_train, y_train, w, lmbd)
        w = w - alpha*grad

    obj_train = objective(X_train, y_train, w, lmbd)
    obj_val = objective(X_val, y_val, w, lmbd)

    return (w, obj_train, obj_val)


def hessian(X, w, lmbd):
    N = X.shape[0]
    M = X.shape[1]
    D = np.zeros((N,N))
    for i in range(N):
        x_i = np.reshape(X[i], (M,1))
        mu_i = np.asscalar(logistic(w.T @ x_i))
        D[i][i] = mu_i * (1-mu_i)
    I = np.identity(M)
    return (X.T @ D) @ X + lmbd*I


def logistic_newtons(X_train, y_train, X_val, y_val, lmbd, num_iter):
    """
    Implement Newton's method on the objective function to find
    weights for L2-regularized logistic regression

    Inputs:
    X_train: (N_train,M) numpy ndarray, N_train examples, M features
    y_train: (N_train,1) numpy ndarray, labels of training data
    X_val: (N_val, M) numpy ndarray, N_val examples, M features
    y_train: (N_val,1) numpy ndarray, labels of validation data
    lmbd: scalar lambda value for regularization
    num_iter: number of iterations to run Newton's method

    Returns:
    w: optimal weights for regression based on gradient descent
    obj_train: objective value for training data (lower is better)
    obj_val: objective value for validation data (lower is better)
    """
    N_train, M = X_train.shape
    N_val = X_val.shape[0]

    # add column of ones in front for bias term
    X_train_ones = np.ones([N_train,1])
    X_train = np.concatenate((X_train_ones, X_train), axis=1)
    X_val_ones = np.ones([N_val,1])
    X_val = np.concatenate((X_val_ones, X_val), axis=1)

    w = np.random.rand(M+1)
    for i in range(num_iter):
        hess = hessian(X_train, w, lmbd)
        grad = gradient(X_train, y_train, w, lmbd)
        w = w - (np.linalg.pinv(hess) @ grad)

    obj_train = objective(X_train, y_train, w, lmbd)
    obj_val = objective(X_val, y_val, w, lmbd)

    return (w, obj_train, obj_val)
