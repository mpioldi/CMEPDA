import numpy as np
import os
import struct
from bitarray import bitarray
from bitarray.util import zeros
import fpzip


def FloatToBitarray(value):
    ba = bitarray()
    ba.frombytes(struct.pack('!d', value))
    return ba

def BitarrayToFloat(ba):
    value = struct.unpack('!d', ba.tobytes())
    return value[0]


def AltDataCompressor(fname, savefile, cut):

    path = './' + fname

    if not os.path.exists(path):
        raise OSError('Requested file not present')

    data = np.loadtxt(fname, skiprows=1)
    data = data[::500]

    a = zeros(64)
    for i in range(64 - cut):
        a.invert(i)
    with np.nditer(data, op_flags=['readwrite']) as it:
        for x in it:
            ba = FloatToBitarray(x)
            ba = ba & a
            x[...] = BitarrayToFloat(ba)

    with open(savefile, "wb") as binary_file:
        binary_file.write(fpzip.compress(data, precision=0, order='C'))

def BinDataCompressor(fname, savefile):

    path = './' + fname

    if not os.path.exists(path):
         raise OSError('Requested file not present')

    data = np.loadtxt(fname)

    with open(savefile, "wb") as binary_file:
        binary_file.write(fpzip.compress(data, precision=0, order='C'))


if __name__ == '__main__':

    complete_compression = 1
    cut = 10

    if complete_compression == 1:
        AltDataCompressor("data.txt", "alt_compr_data.bin", cut)
    else:
        BinDataCompressor("compr_data.txt", "compr_data.bin")



'''
# Test:
fl = -12.0
bitarr = float_to_bitarray(fl)
print(bitarr)
print(type(bitarr))
x = bitarray_to_float(bitarr)
print(type(x))
print(type(fl))
print(x==fl)
print(x)
a = zeros(64)
for i in range(12):
    a.invert(i)
bitarr = bitarr & a
print(bitarr)
'''
