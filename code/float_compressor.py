import numpy as np
import os
import struct
from bitarray import bitarray
from bitarray.util import zeros
import fpzip


def float_to_bitarray(value):
    ba = bitarray()
    ba.frombytes(struct.pack('!d', value))
    return ba

def bitarray_to_float(ba):
    value = struct.unpack('!d', ba.tobytes())
    return value[0]



complete_compression = 1
cut = 10



if complete_compression == 1:
    fname = "data.txt"
    savefile = "alt_compr_data.bin"
else:
    fname = "compr_data.txt"
    savefile = "compr_data.bin"

path = './' + fname

if not os.path.exists(path):
     raise OSError('Requested file not present')

#check if file is .txt

if not fname.endswith(".txt"):
    raise OSError('File is not .txt')

if complete_compression == 1:
    data = np.loadtxt(fname, skiprows=1)
    data = data[::500]
else:
    data = np.loadtxt(fname)

if complete_compression == 1:
    a = zeros(64)
    for i in range(64 - cut):
        a.invert(i)
    with np.nditer(data, op_flags=['readwrite']) as it:
        for x in it:
            ba = float_to_bitarray(x)
            ba = ba & a
            x = bitarray_to_float(ba)

with open(savefile, "wb") as binary_file:
    binary_file.write(fpzip.compress(data, precision=0, order='C'))



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
