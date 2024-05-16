
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import os

from traducers import RootToNumpy, RootTotxt

'''
# Root file data copied to numpy array
data = RootToNumpy("data.root", "tree")
print(data[:100])
data = data[::10]
'''

if os.path.exists("./data.txt"):
    pass
else:
    RootTotxt("data.root", "tree")
    
data = np.loadtxt("data.txt", skiprows=1, ndmin=0)
print(data[:100])
data = data[::1000]

print(f'Input is made of {len(data)} elements \n')

# Dimension of the neural network hidden layers
output_dim = 256
# Weight used for coupling layers regularization
reg = 0.01

''' Custom affine coupling layers for the training of Real NVP scale and translation parameters
there are two output branches, one for s and one for t
each branch has 4 dense layers with relu activation for better training,
while the last layers has linear activation to allow more freedom
for the choice of s and t
l2 regularization penalties (sum of squares) to avoid overfitting
'''
def Coupling(input_shape):
    input = keras.layers.Input(shape=(input_shape,))

    t_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    t_layer_2 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_1)
    t_layer_3 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_2)
    t_layer_4 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_3)
    t_layer_5 = keras.layers.Dense(
        input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_4)

    s_layer_1 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(input)
    s_layer_2 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_1)
    s_layer_3 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_2)
    s_layer_4 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_3)
    s_layer_5 = keras.layers.Dense(
        input_shape, activation="tanh", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_4)

    return keras.Model(inputs=input, outputs=[s_layer_5, t_layer_5])


# Creation of RealNVP class inheriting from keras.Model
class RealNVP(keras.Model):
    #
    def __init__(self, num_coupling_layers):
        super().__init__()

        self.num_coupling_layers = num_coupling_layers

        # Distribution of the latent space: a 15D Gaussian distribution
        self.distribution = tfp.distributions.MultivariateNormalDiag(
            loc=[0.0]*15, scale_diag=[1.0]*15
        )
        ''' Masks used to divide the inputs in two parts:
        one used as inputs of the neural network to generate s and t
        and the other modified by a function of s and t
        the first 5 numbers are always 1s, as they correspond to the 5 (quello che sono, non ricordo...)
        '''
        self.masks_np = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
            * (num_coupling_layers // 3), dtype="float32"
        )
        self.masks = tf.convert_to_tensor(self.masks_np, np.float32)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.layers_list =  [Coupling(20) for i in range(num_coupling_layers)]

    @property
    def metrics(self):
        """List of the model's metrics.

        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    # Implementation of the actual training
    def __call__(self, x, training=True):
        # Starting value of the log determinant of the jacobian
        log_det_inv = 0
        # Direction -1 means training (from our distribution to Gaussian)
        # Direction 1 is the inverse function (from Gaussian to distribution)
        direction = 1
        if training:
            direction = -1
        # Different s and t will be determined for each coupling layer
        # We start from the last coupling layer in the layer_list when going in -1 direction
        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            # This is where the training starts
            s, t = self.layers_list[i](x_masked)
            # s and t must only modify the unmasked values
            s *= reversed_mask
            t *= reversed_mask
            # bijective function applied to unmasked values
            # the gate determines which direction of the function we are applying
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s))
                + x_masked
            )
            # Thanks to the mask, the jacobian is a triangular matrix and its determinant
            # can be easily computed with a simple sum
            log_det_inv += gate * tf.reduce_sum(s, [1])

        return x, log_det_inv

    # Total loss is log likelihood of the normal distribution plus log determinant of the jacobian
    def log_loss(self, x):
        y, logdet = self(x)
        # log likelihood must be computed only on the (i 15 cosi...)
        log_likelihood = self.distribution.log_prob(y[:,5:]) + logdet
        return -tf.reduce_mean(log_likelihood)

    def train_step(self, data):
        with tf.GradientTape() as tape:

            loss = self.log_loss(data)

        # Compute the gradient of the loss with respect to the training variables
        g = tape.gradient(loss, self.trainable_variables)
        # Use gradient to compute new parameters
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}


# Start model training
model = RealNVP(num_coupling_layers=6) # num_coupling_layers should be multiple of 3 (because of mask definition)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))


history = model.fit(
    data, batch_size=256, epochs=300, verbose=2, validation_split=0.2)


# Performance evaluation
plt.figure(figsize=(15, 10))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.legend(["train", "validation"], loc="upper right")
plt.ylabel("loss")
plt.xlabel("epoch")

# From data to latent space: Gaussian distribution (z) generated by our data
z, _ = model(data)

# From latent space to data: data reconstruction (x) from Gaussian (sample)
samples = model.distribution.sample(15000)
#x, _ = model.predict(samples)

# Generate 16 plots in a 4x4 arrangement
f, axes = plt.subplots(4, 4)
f.set_size_inches(20, 15)

# First plot is a 1D Gaussian (for reference)
axes[0, 0].hist(samples[:, 0], bins='auto', color="b")
axes[0, 0].set(title="Generated latent space 1D")
# The other 15 plots are the 15 dimensions of the output form the neural network (should be Gaussians)
for i in range(4):
    for j in range(4):
        if i!=1 or j!=1:
            k = 4*i+j-1
            axes[i-1, j-1].hist(z[:, k], range=(-1,1), bins=100, color="r")
            #axes[i-1, j-1].set(title=f"Inference latent space x_{k}")

plt.show()
