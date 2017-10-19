import numpy as np
import Layers as L
from utils import Progbar, gen_array_from_file
from Loss import binary_cross_entropy, mean_square_error
import optimizers as optim
import pdb
np.random.seed(12345)


# ---- Load the data ---- #
data_train = gen_array_from_file("Data/digitstrain.txt")
train_x = data_train[:, :-1]
data_val = gen_array_from_file("Data/digitsvalid.txt")
val_x = data_val[:, :-1]


class AutoEncoder(object):
    def __init__(self, in_dim, hidden_dim, weight_sharing=False):
        self.in_dim = in_dim
        self.n_dim = hidden_dim
        self.out_dim = in_dim
        self.weight_sharing = weight_sharing
        self.encoder = L.DenseLayer(self.in_dim, self.n_dim)
        self.decoder = L.DenseLayer(self.n_dim, self.out_dim)
        self._layers = ["encoder", "decoder"]
        if self.weight_sharing:
            for l in xrange(len(self._layers) // 2):
                encoder = getattr(self, self._layers[l])
                decoder = getattr(self, self._layers[len(self._layers) - 1 - l])
                assert decoder.params['W'].data.shape == encoder.params['W'].data.transpose().shape
                decoder.params['W'].data = encoder.params['W'].data.transpose()

    def forward(self, input_image):
        """
        The forward pass. Encoder compresses. Decoder Expands
            :param input_image: batch x in_dim : The input image
            :return out_image: batch x in_dim : The reconstructed image
        """
        encoded_im = self.encoder(input_image)
        decoded_im = self.decoder(encoded_im)
        return decoded_im

    def backward(self, output_gradient):
        """
        The backwards pass. Computes the derivatives
            :param output_gradient: batch x in_dim: The gradient w.r.t output_node
            :return gradient: batch x in_dim: The gradient w.r.t entire autoencoder
        """
        grad = output_gradient
        for layer in self._layers[::-1]:
            grad = getattr(self, layer).backward(grad)
        if self.weight_sharing:
            for l in xrange(len(self._layers) // 2):
                encoder = getattr(self, self._layers[l])
                decoder = getattr(self, self._layers[len(self._layers) - 1 - l])
                assert encoder.params['W'].grad.shape == decoder.params['W'].grad.transpose().shape
                gradient = encoder.params['W'].grad + decoder.params['W'].grad.transpose()
                encoder.params['W'].grad = gradient
                decoder.params['W'].grad = gradient.transpose()
        return grad

    def __call__(self, input_image):
        return self.forward(input_image)

    def parameters(self):
        params = {}
        for layer in self._layers:
            params[layer] = getattr(self, layer).parameters()
        return params


IN_DIM = train_x.shape[1]
HIDDEN_DIM = 100

autoencoder = AutoEncoder(IN_DIM, HIDDEN_DIM, weight_sharing=True)

N_EPOCHS = 1000
bar = Progbar(N_EPOCHS)
BATCH_SIZE = 64
lr = 0.1
NOISE = 1.

cross_entropy = binary_cross_entropy()
mse_loss = mean_square_error()
optimizer = optim.SGD(autoencoder.parameters(), lr=0.01)

for epoch in xrange(N_EPOCHS):
    steps = (train_x.shape[0] // BATCH_SIZE) if train_x.shape[0] % BATCH_SIZE == 0 else (train_x.shape[0] // BATCH_SIZE) + 1
    entropy_train_loss = 0.
    mse_train_loss = 0.
    for ix in xrange(steps):
        batch_x = train_x[ix: ix + BATCH_SIZE]
        mask = np.random.binomial(1, NOISE, batch_x.shape)
        input_x = batch_x * mask
        reconstruction = autoencoder(input_x)
        loss = cross_entropy(batch_x, reconstruction)
        entropy_train_loss += loss
        mse_train_loss += mse_loss(batch_x, reconstruction)
        gradient = cross_entropy.grad()
        # gradient = mse_loss.grad()
        autoencoder.backward(gradient)
        optimizer.step()
        # Cleanup
        optimizer.zero_grad()
    entropy_train_loss /= steps
    mse_train_loss /= steps
    val_preds = autoencoder(val_x)
    entropy_val_loss = cross_entropy(val_x, val_preds)
    mse_val_loss = mse_loss(val_x, val_preds)
    bar.update(epoch + 1, values=[("train_entropy", entropy_train_loss),
                                  ("val_entropy", entropy_val_loss),
                                  ("train_mse", mse_train_loss),
                                  ("val_mse", mse_val_loss)])
