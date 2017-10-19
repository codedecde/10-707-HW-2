import numpy as np


class Loss(object):
    """
    Abstract Base class for all lossses
    """
    def __init__(self):
        self._grad = None
        self.eps = np.finfo(float).eps  # For numerical stability

    def __call__(self):
        raise NotImplementedError

    def grad(self):
        assert self._grad is not None, "Error. None gradient found"
        return self._grad


class categorical_cross_entropy(Loss):
    def __init__(self):
        super(categorical_cross_entropy, self).__init__()

    def __call__(self, y, output):
        output += self.eps
        batch_size = y.shape[0]
        loss = -1. * (y * np.log(output))
        self._grad = (output - y)
        loss = np.sum(loss) / batch_size
        return loss


class binary_cross_entropy(Loss):
    def __init__(self):
        super(binary_cross_entropy, self).__init__()

    def __call__(self, y, output):
        output += self.eps
        batch_size = y.shape[0]
        # units = y.shape[1]
        units = 1.
        loss = -1. * ((y * np.log(output)) + ((1. - y) * (np.log(1. - output))))
        self._grad = (output - y) / (output * (1. - output) * units)
        loss = np.sum(loss) / (batch_size * units)
        return loss


class mean_square_error(Loss):
    def __init__(self):
        super(mean_square_error, self).__init__()

    def __call__(self, y, output):
        output += self.eps
        batch_size = y.shape[0]
        loss = 0.5 * (output - y) * (output - y)
        self._grad = (output - y)
        loss = np.sum(loss) / batch_size
        return loss
