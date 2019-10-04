import numpy as np


def affine_forword(X, W, b):
    """
    Affine forword function return result of flatten X multipy W plus b

    Inputs:
        X - shape (N, ...) X can be reshaped to (N, D)
        W - shape (D, C)
        b - shape (D,)
    Outputs:
        out - shape (N, C)
        cache - (X, W, b), that affine_backword needs
    """
    N = X.shape[0]
    flattenX = X.reshape(N, -1)
    out = flattenX.dot(W) + b
    cache = (X, W, b)
    return out, cache


def affine_backword(dout, cache):
    """
    Affine backword function return gradients of X, W and b

    Inputs:
        dout - shape (N, C)
        cache - (X, W, b) from return of affine_forward function
    Outputs:
        dX - shape (N, ...) is same as shape of X in cache
        dW - shape (D, C) is same as shape of W in cache
        db - shape (C,) is same as shape of b in cache
    """
    X, W, b = cache
    N = dout.shape[0]
    flattenX = X.reshape((N, -1))
    dX = dout.dot(W.T)
    dW = flattenX.T.dot(dout)
    db = dout.T.dot(np.ones(N))
    # or db = dout.sum(0)
    return dX.reshape(X.shape), dW, db


def relu_forward(X):
    """
    ReLU activation function

    Input:
        X - any shape array
    Output:
        out - shape is same as shape of X
        cache - X
    """
    out = np.maximum(X, np.zeros_like(X))
    return out, X


def relu_backward(dout, cache):
    """
    Return gradient of ReLU function

    Input:
        dout - shape is same as shape of X in relu_forward
        cache - it is X in relu_forward
    Output:
        dX - shape is same as shape of X in relu_forward
    """
    mask = cache < 0
    dX = dout.copy()
    dX[mask] = 0
    return dX


def softmax_loss(X, y):
    """
    Return softmax value of array X

    Inputs:
        X - shape (N, C)
        y - shape (N,) 0 <= y[i] < C for all 0 <= i < N
    Outous:
        out - shape (N, C)
    """
    N = X.shape[0]
    shifted_X = X - np.max(X, axis=1).reshape(-1, 1)
    p = np.exp(X) /\
        np.sum(np.exp(shifted_X), axis=1).reshape(-1, 1)

    loss = -np.sum(np.log(p[range(N), list(y)])) / N
    dX = p.copy()
    dX[range(N), y] -= 1
    dX /= N
    return loss, dX


def xavier_init(node_in, node_out):
    """
    Xavier initializer, it initializes weights and keeps variance of
    the weight is 1 / sqrt(node_in)

    Input:
        node_in - scalar, shape of initialized weights is (node_in, node_out)
        node_out - scalar, shape of initialized weights is (node_in, node_out)
    Output:
        W - shape (node_in, node_out)
    """
    W = np.random.randn(node_in, node_out) / np.sqrt(node_in)
    return W


def he_init(node_in, node_out):
    """
    He initializer, it initializes weights and keeps variance of
    the weight is 1 / sqrt(node_in/2)

    Input:
        node_in - scalar, shape of initialized weights is (node_in, node_out)
        node_out - scalar, shape of initialized weights is (node_in, node_out)
    Output:
        W - shape (node_in, node_out)
    """
    W = np.random.randn(node_in, node_out) / np.sqrt(node_in/2)
    return W


def zero_init(shapes):
    """
    Zero initializer, it return weights that are zeros

    Input:
        shapes - a list, the shape of weights is same as shapes
    Output:
        W - it is a array of which shape is shapes and values are zero
    """
    W = np.zeros(shapes)
    return W
