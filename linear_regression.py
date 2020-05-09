import numpy as np

def linear_regression(X_train, y_train, X_val, y_val, lmbd):
    """
    Use closed form solution for linear regression to generate weights
    Includes optional parameter lmbd for L2 regularization

    Inputs:
    X_train: (N_train,M) numpy ndarray, N_train examples, M features
    y_train: (N_train,1) numpy ndarray, labels of training data
    X_val: (N_val, M) numpy ndarray, N_val examples, M features
    y_train: (N_val,1) numpy ndarray, labels of validation data
    lmbd: scalar lambda value for regularization

    Returns:
    w: (M+1,1) numpy ndarray, weights based on closed form including bias term
    mse_train: MSE for training data (lower is better)
    mse_val: MSE for validation data (lower is better)
    """
    N_train, M = X_train.shape
    N_val = X_val.shape[0]

    # add column of ones in front to allow for intercept
    X_train_ones = np.ones([N_train,1])
    X_train = np.concatenate((X_train_ones, X_train), axis=1)
    X_val_ones = np.ones([N_val,1])
    X_val = np.concatenate((X_val_ones, X_val), axis=1)

    I = np.identity(M+1)
    first = np.linalg.pinv(X_train.T @ X_train + lmbd*I)
    second = X_train.T @ y_train
    w = first @ second
    y_hat_train = X_train @ w
    mse_train = mse(y_train, y_hat_train)

    y_hat_val = X_val @ w
    mse_val = mse(y_val, y_hat_val)

    return (w, mse_train, mse_val)

def mse(y, y_hat):
    err = y - y_hat
    sqerr = err**2
    return np.mean(sqerr)
