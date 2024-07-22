Usage
=====

Requirements
-----------------

To use NormFlow compression, you will need Python 3, ROOT and the following Python packages:

.. code-block:: console

    import os #only needed to run the tests
    import sys
    import numpy
    import ROOT 
    import tensorflow
    import tensorflow_probability
    import matplotlib.pyplot



Translating from ROOT to numpy an uncompressed database and vice versa
--------------------------------------------------------

.. py:function:: RootToNumpy(name, treename)

    Takes a name.root file, containing a 'treename' tree of the expected shape (a 'Vector'
    branch made of 5 elements arrays of doubles and a 'Matrix' branch of 15 elements arrays
    of doubles) and converts it to an array of 20 elements numpy arrays (shape (,20)).

    :param name: name of the .root file (without extension)
    :type name: str
    :param treename: name of the tree inside the .root file
    :type treename: str
    :return: the data contained in the tree 'treename', in as a numpy array
    :rtype: numpy.array (,20)

.. py:function:: NumpyToRoot(name, treename, myarray)

    Takes an array of 20 elements numpy arrays (shape (,20)) and converts it to a name.root file, 
    containing a 'treename' tree with 2 branches: 'Vector'
    branch made of 5 elements arrays of doubles and 'Matrix' branch of 15 elements arrays
    of doubles.

    :param name: name of the output .root file (without extension)
    :type name: str
    :param treename: name of the tree inside the .root file
    :type treename: str
    :param myarray: the input sequence of 20-elements arrays
    :type myarray: numpy.ndarray
    :raise traducers.ArrayDimensionError: if the input array is not made of 20-elements arrays
    :return: the data contained in the numpy array in a file called name.root, in a tree 'treename'
    :rtype: .root file

The ``array`` parameter should have the right size, otherwise
``RootToNumpy()`` will raise the following exception:

.. py:exception:: traducers.ArrayDimensionError

    Raised if input has not the correct size.

.. _compress:

Compressing data
--------------------------------------------------------

.. py:function:: DataCompressor(fname, n_bins, img=0)

    Takes a fname txt file, containing an array of 20 elements numpy arrays (shape (,20)), and applies a lossy
    compression algorithm that equally divides the data in integer values ('bins') from the interval [0, n_bins)
    by converting them to a normal distribution first, then to a uniform distribution through a cumulative 
    distribution function and dividing the latter into n_bins equally long intervals.

    :param fname: name of the .txt source file
    :type fname: str
    :param n_bits: number of the bins wanted for the compression
    :type n_bits: int
    :param img: if it is 1, images of the distribution of the data at the various steps (before the compression, after converting to gaussians, after converting to uniform and after binning) are produced; defaults to 0 
    :type img: int
    :return: the data contained in fname, compressed
    :rtype: numpy.array (,20)

Decompressing data
--------------------------------------------------------

.. py:function:: DataDecompressor(fname, n_bins, img=0)

    Takes a fname txt file, containing an array of 20 elements numpy arrays (shape (,20)), and applies the inverse of the lossy
    compression of :ref:`DataCompressor <compress>`.

    :param fname: name of the .txt source file 
    :type fname: str
    :param n_bits: number of the bins used for the compression
    :type n_bits: int
    :param img: if it is 1, images of the distribution of the data at the various steps (before the decompression, after converting to gaussians, after reconverting to the original distributions) are produced; defaults to 0 
    :type img: int
    :return: the data contained in fname, decompressed
    :rtype: numpy.array (,20)