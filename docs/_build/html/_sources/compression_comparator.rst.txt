compression_comparator.py
=========================

It uses the functions contained in float_compressor.py, compressor.py and decompressor.py to compress
the experimental data in 2 ways with different parameters.

The first type of compression uses the alt_data_compressor method with different values of cut.

The second type uses data_compressor and bin_data_compressor one after the other, with different values of n_cut.

Finally, the inaccuracy of the compression methods (difference between compressed and original data) is plotted
as a function of the compressed data file size.