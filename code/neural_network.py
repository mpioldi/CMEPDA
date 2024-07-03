import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import tensorflow_probability as tfp

#from traducers import RootToNumpy, RootTotxt

#from traducers import RootTotxt

'''
# Root file data copied to numpy array
data = RootToNumpy("data.root", "tree")
print(data[:100])
data = data[::10]
'''

'''
if os.path.exists("./data.txt"):
    pass
else:
    RootTotxt("data.root", "tree")
'''
#numerb of Coupling layers
layers_number = 42
#file where final gaussian means and variances are saved, necessary for compressor and decompressor
meansfilename = 'means.txt'
# Dimension of the neural network hidden layers
output_dim = 256
# Weight used for coupling layers regularization
reg = 0.01

''' Custom affine coupling layers for the training of Real NVP scale and translation parameters
there are two output branches, one for s and one for t
each branch has 13 dense layers with relu activation for better training,
while the last layers has linear activation to allow more freedom
for the choice of s and t
l2 regularization penalties (sum of squares) to avoid overfitting
'''

#definition of neural network layers
def Coupling(input_shape):
    input = keras.layers.Input(shape=(input_shape,))

    #data normalization layer
    input = keras.layers.BatchNormalization(axis=-1)(input)

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
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_4)
    
    t_layer_6 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_5)
    
    t_layer_7 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_6)
    
    t_layer_8 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_7)
    
    t_layer_9 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_8)
    
    t_layer_10 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_9)
    
    t_layer_11 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_10)
    
    t_layer_12 = keras.layers.Dense(
        input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_11)
    
    t_layer_13 = keras.layers.Dense(
        input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg)
    )(t_layer_12)
    
    

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
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_4)
    
    s_layer_6 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_5)
    
    s_layer_7 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_6)
    
    s_layer_8 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_7)
    
    s_layer_9 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_8)
    
    s_layer_10 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_9)
    
    s_layer_11 = keras.layers.Dense(
        output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_10)
    
    s_layer_12 = keras.layers.Dense(
        input_shape, activation="tanh", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_11)
    
    s_layer_13 = keras.layers.Dense(
        input_shape, activation="tanh", kernel_regularizer=regularizers.l2(reg)
    )(s_layer_12)

    return keras.Model(inputs=input, outputs=[s_layer_13, t_layer_13])


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
        the first 5 numbers are always 1s, as they correspond to
        5 fixed parameters which don't need to be compressed
        '''
        self.masks_np = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]
            * (num_coupling_layers // 2), dtype="float32"
        )
        self.masks = tf.convert_to_tensor(self.masks_np, np.float32)
        #definition of loss tracker
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.layers_list = [Coupling(20) for i in range(num_coupling_layers)]

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
        # log likelihood must be computed only on the 15 unique elements of the matrix
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

    # Used to save final weights if SaveWeights option is selected
    def model_save_weights(self):
        for idx, x in enumerate(self.layers_list):
            x.save_weights(f"weights{idx}.weights.h5")

    # Used to load starting weights if LoadWeights option is selected
    def model_load_weights(self):
        for idx, x in enumerate(self.layers_list):
            x.load_weights(f"weights{idx}.weights.h5")

if __name__ == '__main__':


    SaveWeights = 1
    LoadWeights = 0
    TrainModel = 1


    #loading data
    data = np.loadtxt("data.txt", skiprows=1, ndmin=0)
    print(data[:100])
    data = data[::500]

    #store mean and stdev of data for future conversion
    mus = np.mean(data, axis=0)
    sigmas = np.std(data, axis=0)
    conv_parameters = np.stack([mus, sigmas])
    np.savetxt(meansfilename, conv_parameters, delimiter=' ', newline='\n')

    #normalizing entries
    norm = keras.layers.Normalization(axis=-1)
    norm.adapt(data)
    data = norm(data)

    print(f'Input is made of {len(data)} elements \n')

    # Start model training
    model = RealNVP(num_coupling_layers=layers_number) # num_coupling_layers should be multiple of 2 (because of mask definition)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005))

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=10, min_lr=0.000001)

    # EarlyStopping called in case training isn't effective anymore (no loss reduction)
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=.2, patience=15, restore_best_weights=True)

    if LoadWeights:
        model.model_load_weights()

    if TrainModel:
        history = model.fit(
            data, batch_size=128, epochs=120, verbose=2, validation_split=0.2, callbacks=[reduce_lr, earlystop])


        # Performance evaluation: loss plotting
        plt.figure(figsize=(15, 10))
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.legend(["train", "validation"], loc="upper right")
        plt.ylabel("loss")
        plt.xlabel("epoch")

        plt.savefig('loss.png')

    # From data to latent space: Gaussian distribution (z) generated by our data
    z, _ = model(data)

    print(type(z))

    # Gaussian example: how our final data should look
    gaussian = model.distribution.sample(len(data))

    #plotting 'before' the change of coordinates

    # Generate 16 plots in a 4x4 arrangement
    f1, axes1 = plt.subplots(4, 4)
    f1.set_size_inches(20, 15)

    # First plot is a 1D Gaussian (for reference)
    axes1[0, 0].hist(gaussian[:, 0], bins=500, color="b")
    axes1[0, 0].set(title="Generated latent space 1D")

    # The other 15 plots are the 15 dimensions of the output form the neural network (should be Gaussians)

    for i in range(4):
        #print(f'i={i}')
        for j in range(4):
            k = 4 + i*4 + j
            #print(f'j={j}')
            #print(f'k={k}')
            if k!=4:
                axes1[i, j].hist(data[:, k], range=(-1,1), bins=500, color="r")


                
    plt.savefig('initial.png')

    #plotting 'after' the change of coordinates

    # Generate 16 plots in a 4x4 arrangement
    f, axes = plt.subplots(4, 4)
    f.set_size_inches(20, 15)

    # First plot is a 1D Gaussian (for reference)
    axes[0, 0].hist(gaussian[:, 0], bins=500, color="b")
    axes[0, 0].set(title="Generated latent space 1D")
    # The other 15 plots are the 15 dimensions of the output form the neural network (should be Gaussians)


    for i in range(4):
        #print(f'i={i}')
        for j in range(4):
            k = 4 + i*4 + j
            '''
            print(f'j={j}')
            print(f'k={k}')
            '''
            if k!=4:
                axes[i, j].hist(z[:, k], range=(-4,4), bins=500, color="r")
                axes[i, j].set(title=f"{k}")

                
    plt.savefig('final.png')

    plt.show()

    if SaveWeights:
        model.model_save_weights()



