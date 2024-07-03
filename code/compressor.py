import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
import tensorflow_probability as tfp

from neural_network import RealNVP, layers_number, meansfilename

print('imported without nuisance')

produce_images = 1

#cumulative distribution function of gaussian

def gauss_cdf(x):
    return (1.0 + erf(x / np.sqrt(2.0))) / 2.0

#the name of the file to be compressed should be in the input, in particular it is the second argument (first one is the script name), while the number of bins is the second argument


def DataCompressor(fname, n_bins, img=0):

    #check if file is present

    path = './' + fname

    if not os.path.exists(path):
        raise OSError('Requested file not present')

    #check if file is .txt

    if not fname.endswith(".txt"):
        raise OSError('File is not .txt')
        
    print('started model definition')
        
    #model definition
        
    model = RealNVP(num_coupling_layers=layers_number) #compiling model
    model.compile()
    model.model_load_weights()

    #importing values needed for normalization

    conv_params = np.loadtxt(meansfilename, delimiter=' ')
    mus = conv_params[0, :]
    sigmas = conv_params[1, :]

    #uploading data

    data = np.loadtxt(fname, skiprows=1)
    data = data[::100]

    print('data imported')

    #normalization

    norm = keras.layers.Normalization(axis=-1, mean=mus, variance=sigmas**2)
    data = norm(data)

    print('data normalized')

    # From data to latent space: Gaussian distribution (z) generated by our data
    z, _ = model(data)

    print('data converted')

    #convert to uniform

    #z_unif = z
    z_unif = z.numpy()
    z_unif[:, 5:] = float(n_bins)*gauss_cdf(z[:, 5:])
    z_unif = tf.convert_to_tensor(z_unif)

    #compress data

    #z_int = z
    z_int = z.numpy()
    z_int[:, 5:] = np.floor(z_unif[:, 5:])
    #z_compr = z
    z_compr = z.numpy()
    z_compr[:, 5:] = z_int.astype(int)[:, 5:]
    z_compr = tf.convert_to_tensor(z_compr)
    
    #plots, if requested:
    if img == 1:
    
        # From latent space to data: data reconstruction (x) from Gaussian (sample)
        samples = model.distribution.sample(len(data))
        #x, _ = model.predict(samples)

        #plotting 'before' the change of coordinates

        # Generate 16 plots in a 4x4 arrangement
        f, axes = plt.subplots(4, 4)
        f.set_size_inches(20, 15)

        # First plot is a 1D Gaussian (for reference)
        axes[0, 0].hist(samples[:, 0], bins=500, color="b")
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
                    axes[i, j].hist(data[:, k], range=(-0.5,0.5), bins=500, color="r")
                    
        plt.savefig('precompressed.png')

    
        # Generate 16 plots in a 4x4 arrangement after change of coordinates
        f1, axes1 = plt.subplots(4, 4)
        f1.set_size_inches(20, 15)

        # First plot is a 1D Gaussian (for reference)
        axes1[0, 0].hist(samples[:, 0], bins=500, color="b")
        axes1[0, 0].set(title="Generated latent space 1D")
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
                    axes1[i, j].hist(z[:, k], range=(-4,4), bins=500, color="r")
                    axes1[i, j].set(title=f"{k}")

                    
        plt.savefig('gaussians.png')

    
        # Generate 16 plots in a 4x4 arrangement after switching to uniform
        f2, axes2 = plt.subplots(4, 4)
        f2.set_size_inches(20, 15)

        for i in range(4):
            #print(f'i={i}')
            for j in range(4):
                k = 4 + i*4 + j
                '''
                print(f'j={j}')
                print(f'k={k}')
                '''
                if k!=4:
                    axes2[i, j].hist(z_unif[:, k], range=(0, int(n_bins)), bins=500, color="r")
                    axes2[i, j].set(title=f"{k}")

                    
        plt.savefig('uniform.png')

    
        # Generate 16 plots in a 4x4 arrangement for z_compr
        f3, axes3 = plt.subplots(4, 4)
        f3.set_size_inches(20, 15)

        for i in range(4):
            #print(f'i={i}')
            for j in range(4):
                k = 4 + i*4 + j
                '''
                print(f'j={j}')
                print(f'k={k}')
                '''
                if k!=4:
                    axes3[i, j].hist(z_compr[:, k], range=(0, int(n_bins)), bins=500, color="r")
                    axes3[i, j].set(title=f"{k}")

        plt.savefig('compressed.png')

        #show graphs

        plt.show()
        
    return z_compr

if __name__ == '__main__':

    name = "data.txt"
    n = 2048
    result = DataCompressor(name, n, img=produce_images)
    np.savetxt("compr_data.txt", result, delimiter=' ', newline='\n', header='')

    
    
    
    
