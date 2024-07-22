
import numpy as np
import os
import struct
from bitarray import bitarray
from bitarray.util import zeros
import fpzip


# transform a float value into a bitarray
def float_to_bitarray(value):
    # create empty bitarray
    ba = bitarray()
    # transform the float into a byte array and then into a bitarray
    ba.frombytes(struct.pack('!d', value))
    return ba


# transform a bitarray back into a float
def bitarray_to_float(ba):
    # transform bitarray into a byte array and then into an array containing the corresponding float
    value = struct.unpack('!d', ba.tobytes())
    return value[0]


''' alternative compression method with respect to the one involving the neural network
it transforms the floats of a dataset into bitarrays,
then cuts a part of their mantissa by turning it into zeroes,
turns the dataset elements back into floats
and finally saves the data into a .bit file after compressing it with the fpzip algorithm

parameters
----------
fname: string
    name of the original data file
savefile: string
    name of the file where the compressed data will be stored
cut: integer
    number of mantissa bits that will be turned into zeroes
'''
def alt_data_compressor(fname, savefile, cut):

    # verify the presence of the file
    path = './' + fname
    if not os.path.exists(path):
        raise OSError('Requested file not present')

    # load data to be compressed
    data = np.loadtxt(fname, skiprows=1)
    data = data[::100]

    # create a bitarray with the dimension of a float (64) and a number of 0 bits at the end equal to the value of cut
    a = zeros(64)
    for i in range(64 - cut):
        a.invert(i)
    # iterate over the dataset and turn the mantissa bits into zeroes, according to the value of cut
    with np.nditer(data, op_flags=['readwrite']) as it:
        for x in it:
            ba = float_to_bitarray(x)
            ba = ba & a
            x[...] = bitarray_to_float(ba)

    # compress data into a .bit file with the fpzip algorithm
    savepath = './alt_compr_data/' + savefile
    with open(savepath, "wb") as binary_file:
        binary_file.write(fpzip.compress(data, precision=0, order='C'))


''' compression method to be used on the data previously compressed with compressor.py
it turns the floats into bitarrays and compresses them with the fpzip algorithm
making it possible to compare the normalizing flow compression with the mantissa cut compression described above

parameters
----------
fname: string
    name of the original data file
savefile: string
    name of the file where the compressed data will be stored
'''

def bin_data_compressor(fname, savefile):

    # verify the presence of the file
    path = './compr_data/' + fname
    if not os.path.exists(path):
        raise OSError('Requested file not present')

    # load data to be compressed
    data = np.loadtxt(path)

    # compress data into a .bit file with the fpzip algorithm
    savepath = './compr_data/' + savefile
    with open(savepath, "wb") as binary_file:
        binary_file.write(fpzip.compress(data, precision=0, order='C'))


if __name__ == '__main__':

    # if true, use the bin_data_compressor method, otherwise use alt_data_compressor
    complete_compression = 1
    cut = 10

    if complete_compression == 1:
        alt_data_compressor("data.txt", "alt_compr_data.bin", cut)
    else:
        bin_data_compressor("compr_data.txt", "compr_data.bin")

