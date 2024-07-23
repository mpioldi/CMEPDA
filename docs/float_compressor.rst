float_compressor.py
=====

Contains the functions used to compress data into binary .bit files

.. py:function:: alt_data_compressor(fname, savefile, cut):

    Alternative compression method with respect to the one involving the neural network.
    It transforms the floats of a dataset into bitarrays,
    then cuts a part of their mantissa by turning it into zeroes,
    turns the dataset elements back into floats,
    and finally saves the data into a .bit file after compressing it with the fpzip algorithm.

    :param fname: name of the file containing the data to be compressed
    :type fname: str
    :param savefile: name of the file where the compressed data will be stored
    :type savefile: str
    :param cut: number of mantissa bits that will be turned into zeroes
    :type cut: int
    :raise OSError: if file does not exist

.. py:function:: bin_data_compressor(fname, savefile):

    Compression method to be used on the data previously compressed with compressor.py.
    It turns the floats into bitarrays and compresses them with the fpzip algorithm,
    making it possible to compare the normalizing flow compression with the mantissa cut compression described above.

    :param fname: name of the file containing the data to be compressed
    :type fname: str
    :param savefile: name of the file where the compressed data will be stored
    :type savefile: str
    :raise OSError: if file does not exist

