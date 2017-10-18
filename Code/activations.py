import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def sigmoid_grad(x):
    sigmoid_grad = sigmoid(x)
    sigmoid_grad = sigmoid_grad * (1. - sigmoid_grad)
    return sigmoid_grad


def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    return (1. - (tanh(x) ** 2))


def relu(x):
    return np.maximum(0., x)


def relu_grad(x):
    gradient = np.ones(x.shape)
    gradient[x < 0] = 0
    return gradient


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1).reshape(x.shape[0], 1))
    return e_x / np.sum(e_x, axis=1).reshape(e_x.shape[0], 1)


def softmax_grad(x):
    # This is a special case. It is easier to precompute the gradient wrt the activation and send it below
    return 1.


def linear(x):
    return x


def linear_grad(x):
    return 1.


def leaky_relu(x, alpha=0.01):
    mask = np.ones(x.shape)
    mask[x < 0] = alpha
    return x * mask


def leaky_relu_grad(x, alpha=0.01):
    mask = np.ones(x.shape)
    mask[x < 0] = alpha
    return mask
