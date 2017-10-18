import numpy as np
import pdb
import activations as A
from utils import Progbar, gen_array_from_file
import cPickle as cp
import matplotlib
from Layers import Variable
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# LOAD THE DATA ###
data_train = gen_array_from_file("Data/digitstrain.txt")
train_x = data_train[:, :-1]
data_val = gen_array_from_file("Data/digitsvalid.txt")
val_x = data_val[:, :-1]
SAMPLE = False


class RBM(object):
    def __init__(self, in_dim, hidden_dim):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        # Parameters
        # _init = np.sqrt(6) / np.sqrt(self.hidden_dim + self.in_dim)
        # W = np.random.uniform(low=-_init, high=_init, size=(self.in_dim, self.hidden_dim))
        W = np.random.normal(loc=0., scale=0.1, size=(self.in_dim, self.hidden_dim))
        b = np.ones((self.hidden_dim, ))
        c = np.ones((self.in_dim, ))
        self.W = Variable(W)  # in_dim x hidden_dim
        self.b = Variable(b)  # hidden_dim,
        self.c = Variable(c)  # in_dim,
        self.parameters = {"W": self.W, "b": self.b, "c": self.c}

    def _h_given_x(self, x):
        """
        Gives the probability vector of each hidden being 1 given x
            :param x: batch x in_dim: The observed x
            :return h: batch x hidden_dim: The probability of each h = 1 given x
        """
        return A.sigmoid(np.dot(x, self.W.data) + self.b.data)

    def _x_given_h(self, h):
        """
        Gives the probability vector of each x being 1 given h
            :param h: batch x hidden_dim: The observed h
            :return x: batch x in_dim: The probability of each x = 1 given h
        """
        return A.sigmoid(np.dot(h, self.W.data.transpose()) + self.c.data)

    def compute_grad(self, x, x_tilde):
        """
        Computes E_{h|x}(\nabla_{\theta} -log(p(h,x))) - E_{x, h}(\nabla_{\theta} -log(p(h,x)))
        x_tilde is the MCMC sample to approximate the E_{x, h}
        Updates W, b and c
            :param x: batch x in_dim: The input image
            :param x_tilde: batch x in_dim: The image sampled from the model distribution. Obtained from a Gibbs sampler
        """
        batch_size = x.shape[0]
        h_given_x = self._h_given_x(x)
        h_given_x_tilde = self._h_given_x(x_tilde)
        grad_E_h_given_x = np.dot(x.transpose(), h_given_x)
        grad_E_h_x = np.dot(x.transpose(), h_given_x_tilde)
        self.W.grad = -1. * (grad_E_h_given_x - grad_E_h_x) / batch_size
        self.b.grad = -1. * np.sum((h_given_x - h_given_x_tilde), 0) / batch_size
        self.c.grad = -1. * np.sum((x - x_tilde), 0) / batch_size

    def sample(self, probs):
        """
        Samples from a bernoulli distribution, defined by probs
            :param probs: batch x dim: The probability matrix
            :return sample: batch x dim: The sampled version
        """
        sample = np.random.uniform(low=0., high=1., size=probs.shape)
        sample = 1. * (sample < probs)
        return sample

    def gibbs_sampler(self, x, k):
        """
        Gibbs sampler. Samples hidden state, and then the image. Does it k times.
            :param x: batch x in_dim: The true image
            :param k: int: The number of steps to carry out
            :return x_tilde: batch x in_dim: The probability of the sampled image
        """
        x_tilde = x
        for ix in xrange(k):
            # Sample h
            h = self._h_given_x(x_tilde)
            if SAMPLE:
                h = self.sample(h)
            x_tilde = self._x_given_h(h)
            if ix != k - 1 and SAMPLE:
                x_tilde = self.sample(x_tilde)
        return x_tilde


class SGD(object):
    def __init__(self, parameters, lr=0.01):
        # TODO : Add momentum
        self.params = parameters
        self.lr = lr

    def step(self):
        for param in self.params:
            if type(param) == dict:
                # params is a list of dictionaries. param is a dictionary
                for key in param:
                    param[key].data -= self.lr * param[key].grad
            else:
                # params is a dictionary of parameters. param is the key
                self.params[param].data -= self.lr * self.params[param].grad

    def zero_grad(self):
        for param in self.params:
            if type(param) == dict:
                # params is a list of dictionaries. param is a dictionary
                for key in param:
                    param[key].grad = np.zeros(param[key].data.shape)
            else:
                # params is a dictionary of parameters. param is the key
                self.params[param].grad = np.zeros(self.params[param].data.shape)


def cross_entropy(true_image, sampled_image):
    """
    Computes the cross entropy loss between the true image and the sampled image
        :param true_image: batch x in_dim: The true image
        :param sampled_image: batch x in_dim: The sampled image
        :return loss: float: The average cross_entropy loss
    """
    loss = -1. * np.sum((true_image * np.log(sampled_image + np.finfo(float).eps) + ((1. - true_image) * np.log(1. - sampled_image + np.finfo(float).eps)))) / (true_image.shape[0] * true_image.shape[1])
    return loss


def plot(array, filename, epoch):
    cols = int(np.sqrt(array.shape[0]))
    rows = (array.shape[0] // cols) if (array.shape[0] % cols == 0) else (array.shape[0] // cols) + 1
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    fig.suptitle('Images After {} epochs'.format(epoch))
    index = 0
    for row_ix, row in enumerate(axes):
        for col_ix, ax in enumerate(row):
            if index >= array.shape[0]:
                ax.imshow(np.zeros((28, 28)), cmap='gray')
            else:
                ax.imshow(array[index, :].reshape((28, 28)), cmap='gray')
            index += 1
            ax.set_xticks([])
            ax.set_yticks([])
    plt.savefig(filename)
    plt.close()


HIDDEN_DIM = 200
IN_DIM = train_x.shape[1]
N_EPOCHS = 200
BATCH_SIZE = 64
prev_sample = None
PLOT_AFTER = 20
PLOT_GIBBS_STEPS = 100
k = 5
lr = 0.01

rbm = RBM(IN_DIM, HIDDEN_DIM)
optimizer = SGD(rbm.parameters, lr=lr)
batch_ix = 0
bar = Progbar(N_EPOCHS)
for epoch in xrange(N_EPOCHS):
    steps = train_x.shape[0] // BATCH_SIZE if (train_x.shape[0] % BATCH_SIZE == 0) else (train_x.shape[0] // BATCH_SIZE) + 1
    avg_train_loss = 0.
    for step in xrange(steps):
        batch_x = train_x[batch_ix: batch_ix + BATCH_SIZE, :]
        batch_ix += BATCH_SIZE
        if batch_ix >= train_x.shape[0]:
            batch_ix = 0
        model_probs = rbm.gibbs_sampler(batch_x, k)
        if SAMPLE:
            model_sample = rbm.sample(model_probs)
        else:
            model_sample = model_probs
        prev_sample = model_sample
        train_loss = cross_entropy(batch_x, model_probs)
        avg_train_loss += train_loss
        rbm.compute_grad(batch_x, model_sample)
        optimizer.step()
        # Clean Up
        optimizer.zero_grad()
    avg_train_loss /= steps
    val_loss = cross_entropy(val_x, rbm.gibbs_sampler(val_x, k))
    bar.update(epoch + 1, values=[("train_loss", avg_train_loss), ("val_loss", val_loss)])
    if (epoch + 1) % PLOT_AFTER == 0:
        # index = range(train_x.shape[0])
        # random.shuffle(index)
        # random_train = train_x[index, :][:16,:]
        random_train = np.random.normal(0, 1., (16, 784))
        sample = rbm.gibbs_sampler(random_train, PLOT_GIBBS_STEPS)
        if SAMPLE:
            sample = rbm.sample(sample)
        plot(sample, "Plots/Image_Epoch_{}.png".format(epoch + 1), epoch + 1)
# Save the weights
save_filename = "Models/weights_hdim_%d_epochs_%d_val_%.4f_k_%d.pkl" % (HIDDEN_DIM, N_EPOCHS, val_loss, k)
cp.dump(rbm, open(save_filename, "wb"))
