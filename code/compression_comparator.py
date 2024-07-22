
import numpy as np
import os
import matplotlib.pyplot as plt
import fpzip

from float_compressor import alt_data_compressor, bin_data_compressor
from compressor import data_compressor
from decompressor import data_decompressor


# extracts the data of the .bin file "fname" compressed with the fpzip algorithm and returns it as a numpy array
def bit_decompressor(fname):

    # verify the presence of the file
    path = './' + fname
    if not os.path.exists(path):
        raise OSError('Requested file not present')

    # read the file
    with open(fname, "rb") as file:
        compressed_bytes = file.read()

    # extract the file data into a numpy array
    data = fpzip.decompress(compressed_bytes, order='C')

    # return the decompressed data, which is stored in a multidimensional numpy array with 2 extra dimensions
    # because of the way the fpzip.decompress method works
    return data[0, 0]


# compare compressed and uncompressed data to check how much the compressed data differs from the original data
def comparator(uncompressed, compressed):

    # create an array where every element is the relative difference between compressed and uncompressed floats
    deltas = np.zeros(np.shape(uncompressed))
    deltas[:, 5:] = (compressed[:, 5:] - uncompressed[:, 5:]) / np.abs(uncompressed[:, 5:])
    # calculate the standard deviation of the deltas for each of the 15 elements of the data matrix,
    # excluding the compressed data that deviates too much from the original
    deltas_wo_tails = np.ma.masked_where((deltas < -0.1) | (deltas > 0.1), deltas)
    widths = np.std(deltas_wo_tails[:, 5:], axis=0)

    # return the mean and the standard deviation of the 15 widths,
    # which will be used in the final graph to determine the inaccuracy of the compression methods
    return np.mean(widths), np.std(widths)


# name of the original data file
fname = "data.txt"
path = './' + fname

# verify the presence of the file
if not os.path.exists(path):
    raise OSError('Requested file not present')

# load the data
data = np.loadtxt(fname, skiprows=1)
data = data[::100]

# these arrays will store the values related to the compression accuracy of alt_data_compressor for different cut values

# array of the file size values of the different batches of compressed data
alt_compr_size = np.zeros(20)
# arrays of the comparator method "width" means and stds for the different batches of compressed data
alt_compr_inaccuracy = np.zeros(20)
alt_compr_inaccuracy_std = np.zeros(20)
# array of the annotations used in the final plot
alt_compr_annotation = np.zeros(20)

n = 0

# compress the original data with the alt_data_compressor method with different cut values
for cut in list(range(0, 45, 4)) + list(range(45, 53, 1)):
    # names of the files where the compressed data will be stored
    savefile = f"alt_compr_data{cut}.bin"
    path = './' + savefile

    # if the data hasn't been compressed with a certain cut value, it will be compressed
    if not os.path.exists(path):
        alt_data_compressor("data.txt", savefile, cut)

    # save the size of the compressed data file
    alt_compr_size[n] = os.path.getsize(path)
    # decompress the data
    alt_decompr_data = bit_decompressor(savefile)
    # save the inaccuracy of the compressed data with the comparator method
    alt_compr_inaccuracy[n], alt_compr_inaccuracy_std[n] = comparator(data, alt_decompr_data)
    # save the cut value to be used as an annotation in the final plot
    alt_compr_annotation[n] = cut
    
    n += 1

# these arrays will store the values related to the compression accuracy of the data_compressor method
# for different numbers of bins

# array of the file size values of the different batches of compressed data
compr_size = np.zeros(17)
# arrays of the comparator method "width" means and stds for the different batches of compressed data
compr_accuracy = np.zeros(17)
compr_accuracy_std = np.zeros(17)
# array of the annotations used in the final plot
compr_annotation = np.zeros(17)

# compress the original data with data_compressor with different n_bins values and then with bin_data_compressor
for n_bins in range(0, 4097, 256):

    # names of the files where the data_compressor compressed data will be stored
    compr_txt_savefile = f"compr_data{n_bins}.txt"
    # names of the files where the bin_data_compressor compressed data will be stored
    compr_bin_savefile = f"compr_data{n_bins}.bin"
    # names of the files where the decompressed data will be stored
    decompr_savefile = f"decompr_data{n_bins}.txt"

    ctpath = './' + compr_txt_savefile
    cbpath = './' + compr_bin_savefile
    dpath = './' + decompr_savefile

    # if the data hasn't been compressed with data_compressor for a certain n_bins value, it will be compressed
    if not os.path.exists(ctpath):
        result = data_compressor("data.txt", n_bins)
        np.savetxt(compr_txt_savefile, result, delimiter=' ', newline='\n', header='')

    # if the data compressed by data_compressor hasn't been compressed with bin_data_compressor, it will be compressed
    if not os.path.exists(cbpath):
        bin_data_compressor(compr_txt_savefile, compr_bin_savefile)

    # save the size of the compressed data file
    n = int(n_bins / 256)
    compr_size[n] = os.path.getsize(cbpath)

    # if the data hasn't been decompressed yet, it will be decompressed
    if not os.path.exists(dpath):
        result = data_decompressor(ctpath, n_bins)
        np.savetxt(decompr_savefile, result, delimiter=' ', newline='\n', header='')

    # load the decompressed data
    decompr_data = np.loadtxt(decompr_savefile)
    # save the inaccuracy of the compressed data with the comparator method
    compr_accuracy[n], compr_accuracy_std[n] = comparator(data, decompr_data)
    # save the n_bins value to be used as an annotation in the final plot
    compr_annotation[n] = n_bins


# create a plot of the compression inaccuracy as a function of the compressed data file size
plt.figure(figsize=(15, 10))

# data_compressor data
plt.errorbar(compr_size, compr_accuracy, compr_accuracy_std, fmt='.')
# annotate the plot points
'''
for i, (xi, yi) in enumerate(zip(compr_size, compr_accuracy)):
    plt.annotate(f'{int(compr_annotation[i])}', (xi, yi), textcoords="offset points", xytext=(0,10), ha='center')
'''

# alt_data_compressor data
plt.errorbar(alt_compr_size, alt_compr_inaccuracy, alt_compr_inaccuracy_std, fmt='.')
# annotate the plot points
'''
for i, (xi, yi) in enumerate(zip(alt_compr_size, alt_compr_accuracy)):
    plt.annotate(f'{int(alt_compr_annotation[i])}', (xi, yi), textcoords="offset points", xytext=(s0,10), ha='center')
'''

plt.title("compression accuracy")
plt.legend(["NVP compr.", "Alt. compr."], loc="upper right")
plt.ylabel("accuracy")
plt.xlabel("filesize")

plt.savefig('compr_accuracy.png')

plt.show()

