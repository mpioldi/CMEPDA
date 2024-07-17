import numpy as np
import os
import matplotlib.pyplot as plt
import fpzip

from float_compressor import AltDataCompressor, BinDataCompressor
from compressor import DataCompressor
from decompressor import DataDecompressor


def bit_decompressor(fname):
    path = './' + fname

    if not os.path.exists(path):
        raise OSError('Requested file not present')

    with open(fname, "rb") as file:
        compressed_bytes = file.read()

    data = fpzip.decompress(compressed_bytes, order='C')

    return data

def comparator(uncompressed, compressed):

    deltas = np.zeros(np.shape(uncompressed))
    deltas[:, 5:] = (compressed[:, 5:] - uncompressed[:, 5:]) / np.abs(uncompressed[:, 5:])
    deltas_wo_tails = np.ma.masked_where((deltas < -0.1) | (deltas > 0.1), deltas)
    widths = np.std(deltas_wo_tails[:, 5:], axis=0)

    return np.mean(widths), np.std(widths)


fname = "data.txt"
path = './' + fname

if not os.path.exists(path):
    raise OSError('Requested file not present')

data = np.loadtxt(fname, skiprows=1)
data = data[::500]


alt_compr_size = np.zeros(14)
alt_compr_accuracy = np.zeros(14)
alt_compr_accuracy_std = np.zeros(14)

for cut in range(0, 53, 4):

    savefile = f"alt_compr_data{cut}.bin"
    path = './' + savefile

    if not os.path.exists(path):
        AltDataCompressor("data.txt", savefile, cut)

    alt_compr_size[cut/4] = os.path.getsize(path)

    alt_decompr_data = bit_decompressor(savefile)
    alt_compr_accuracy[cut/4], alt_compr_accuracy_std[cut/4] = comparator(data, alt_decompr_data)


compr_size = np.zeros(17)
compr_accuracy = np.zeros(17)
compr_accuracy_std = np.zeros(17)

for n_bins in range(0, 4097, 256):

    compr_txt_savefile = f"compr_data{n_bins}.txt"
    compr_bin_savefile = f"compr_data{n_bins}.bin"
    decompr_savefile = f"decompr_data{n_bins}.txt"

    ctpath = './' + compr_txt_savefile
    cbpath = './' + compr_bin_savefile
    dpath = './' + decompr_savefile

    if not os.path.exists(ctpath):
        result = DataCompressor("data.txt", n_bins)
        np.savetxt(compr_txt_savefile, result, delimiter=' ', newline='\n', header='')

    if not os.path.exists(cbpath):
        BinDataCompressor(compr_txt_savefile, compr_bin_savefile)

    compr_size[n_bins / 256] = os.path.getsize(cbpath)

    if not os.path.exists(dpath):
        result = DataDecompressor(ctpath, n_bins)
        np.savetxt(decompr_savefile, result, delimiter=' ', newline='\n', header='')

    decompr_data = np.loadtxt(decompr_savefile)
    compr_accuracy[n_bins / 256], compr_accuracy_std[n_bins / 256] = comparator(data, decompr_data)


plt.figure(figsize=(15, 10))
plt.errorbar(compr_size, compr_accuracy, compr_accuracy_std)
plt.errorbar(alt_compr_size, alt_compr_accuracy, alt_compr_accuracy_std)
plt.title("compression accuracy")
plt.legend(["NVP compr.", "Alt. compr."], loc="upper right")
plt.ylabel("accuracy")
plt.xlabel("filesize")

plt.savefig('compr_accuracy.png')


