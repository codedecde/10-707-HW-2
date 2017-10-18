import numpy as np
import Layers as L
from utils import Progbar, gen_array_from_file
from Loss import binary_cross_entropy
import optimizers as optim


# ---- Load the data ---- #
data_train = gen_array_from_file("Data/digitstrain.txt")
train_x = data_train[:, :-1]
data_val = gen_array_from_file("Data/digitsvalid.txt")
val_x = data_val[:, :-1]


class AutoEncoder(object):
    def __init__(self, in_dim, hidden_dim):
        self.in_dim = in_dim
        self.n_dim = hidden_dim
        self.out_dim = in_dim
        self.encoder = L.DenseLayer(self.in_dim, self.n_dim)
        self.decoder = L.DenseLayer(self.n_dim, self.out_dim)
        self._layers = ["encoder", "decoder"]

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

autoencoder = AutoEncoder(IN_DIM, HIDDEN_DIM)

N_EPOCHS = 1000
bar = Progbar(N_EPOCHS)
BATCH_SIZE = 64

cross_entropy = binary_cross_entropy()
optimizer = optim.SGD(autoencoder.parameters(), BATCH_SIZE)

for epoch in xrange(N_EPOCHS):
    steps = (train_x.shape[0] // BATCH_SIZE) if train_x.shape[0] % BATCH_SIZE == 0 else (train_x.shape[0] // BATCH_SIZE) + 1
    avg_train_loss = 0.
    for ix in xrange(steps):
        batch_x = train_x[ix: ix + BATCH_SIZE]
        # TODO: Add noise
        reconstruction = autoencoder(batch_x)
        loss = cross_entropy(batch_x, reconstruction)
        avg_train_loss += loss
        autoencoder.backward(cross_entropy.grad())
        optimizer.step()
        # Cleanup
        optimizer.zero_grad()
    avg_train_loss /= steps
    val_preds = autoencoder(val_x)
    val_loss = cross_entropy(val_x, val_preds)
    bar.update(epoch + 1, values=[("train_loss", avg_train_loss), ("val_loss", val_loss)])
