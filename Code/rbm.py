import numpy as np
import pdb
import activations as A
from utils import Progbar, gen_array_from_file, plot
import cPickle as cp
from Layers import Variable
from Loss import binary_cross_entropy
import argparse
import sys
from optimizers import SGD


SAMPLE = True


def get_arguments():
    def check_boolean(args, attr_name):
        assert hasattr(args, attr_name), "%s not found in parser" % (attr_name)
        bool_set = set(["true", "false"])
        args_value = getattr(args, attr_name)
        args_value = args_value.lower()
        assert args_value in bool_set, "Boolean argument required for attribute %s" % (attr_name)
        args_value = False if args_value == "false" else True
        setattr(args, attr_name, args_value)
        return args
    parser = argparse.ArgumentParser(description='Restricted Boltzmann machine')
    parser.add_argument('-n_hidden', action="store", default=200, dest="n_hidden", type=int)
    parser.add_argument('-batch', action="store", default=64, dest="batch_size", type=int)
    parser.add_argument('-l2', action="store", default=0.000, dest="l2", type=float)
    parser.add_argument('-lr', action="store", default=0.01, dest="lr", type=float)
    parser.add_argument('-momentum', action="store", default=0.0, dest="momentum", type=float)
    parser.add_argument('-n_epochs', action="store", default=50, dest="n_epochs", type=int)
    parser.add_argument('-plot_after', action="store", default=5, dest="plot_after", type=int)
    parser.add_argument('-cd_k', action="store", default=1, dest="cd_k", type=int)
    parser.add_argument('-gibbs_steps', action='store', default=1000, dest='gibbs_steps', type=int)
    # Using strings as a proxy for boolean flags. Checks happen later
    args = parser.parse_args(sys.argv[1:])
    # Checks for the boolean flags
    return args


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
        self.params = {'rbm_layer': {"W": self.W, "b": self.b, "c": self.c}}

    def parameters(self):
        return self.params

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

    def backward(self, x, x_tilde):
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

    def gibbs_sampler(self, x, k, return_probs=True):
        """
        Gibbs sampler. Samples hidden state, and then the image. Does it k times.
            :param x: batch x in_dim: The true image
            :param k: int: The number of steps to carry out
            :param return_probs: bool: Return a sample or the probability distribution as the result
            :return x_: batch x in_dim: The output image, as controlled by return_probs
        """
        x_tilde = x
        x_probs = None
        x_sample = None
        for ix in xrange(k):
            # Sample h
            h_probs = self._h_given_x(x_tilde)
            h_sample = self.sample(h_probs)
            h = h_sample if SAMPLE else h_probs
            x_probs = self._x_given_h(h)
            x_sample = self.sample(x_probs)
            x_tilde = x_sample if SAMPLE else x_probs
        ret_img = x_probs if return_probs else x_sample
        return ret_img


if __name__ == "__main__":
    # LOAD THE DATA ###
    data_train = gen_array_from_file("Data/digitstrain.txt")
    train_x = data_train[:, :-1]
    data_val = gen_array_from_file("Data/digitsvalid.txt")
    val_x = data_val[:, :-1]
    args = get_arguments()
    IN_DIM = train_x.shape[1]
    BATCH_SIZE = args.batch_size
    k = args.cd_k
    lr = args.lr
    cross_entropy = binary_cross_entropy()
    rbm = RBM(IN_DIM, args.n_hidden)
    optimizer = SGD(rbm.parameters(), lr=lr, l2=args.l2, momentum=args.momentum)
    batch_ix = 0
    bar = Progbar(args.n_epochs)
    for epoch in xrange(args.n_epochs):
        steps = train_x.shape[0] // BATCH_SIZE if (train_x.shape[0] % BATCH_SIZE == 0) else (train_x.shape[0] // BATCH_SIZE) + 1
        avg_train_loss = 0.
        for step in xrange(steps):
            batch_x = train_x[batch_ix: batch_ix + BATCH_SIZE, :]
            batch_ix += BATCH_SIZE
            if batch_ix >= train_x.shape[0]:
                batch_ix = 0
            model_probs = rbm.gibbs_sampler(batch_x, k, return_probs=True)
            model_sample = rbm.sample(model_probs) if SAMPLE else model_probs
            train_loss = cross_entropy(batch_x, model_probs)
            avg_train_loss += train_loss
            rbm.backward(batch_x, model_sample)
            optimizer.step()
            # Clean Up
            optimizer.zero_grad()
        avg_train_loss /= steps
        val_sample = rbm.gibbs_sampler(val_x, k, return_probs=True)
        val_loss = cross_entropy(val_x, val_sample)
        bar.update(epoch + 1, values=[("train_loss", avg_train_loss), ("val_loss", val_loss)])
        if (epoch + 1) % args.plot_after == 0:
            # index = range(train_x.shape[0])
            # random.shuffle(index)
            # random_train = train_x[index, :][:16,:]
            random_train = np.random.uniform(0, 1., (100, 784))
            sample = rbm.gibbs_sampler(random_train, args.gibbs_steps, return_probs=True)
            plot(sample, "Plots/Image_Epoch_{}.png".format(format(epoch + 1, '03')), epoch + 1)
    # Save the weights
    save_filename = "Models/weights_hdim_%d_epochs_%d_val_%.4f_k_%d.pkl" % (args.n_hidden, args.n_epochs, val_loss, k)
    cp.dump(rbm, open(save_filename, "wb"))
