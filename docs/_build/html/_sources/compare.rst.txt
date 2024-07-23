compare.py
==========

It requires to have run compressor.py and decompressor.py (and all their dependencies).
It uploads a subset of the 'data.txt' dataset (by np.loadtxt("data.txt")[::100]), evaluates
the relative error committed on the data after a compression and a decompression and plots histograms showing the
distribution of the committed errors.