decompressor.py
===============

Requirements:

.. code-block:: console

    from neural_network import RealNVP, layers_number, meansfilename

data.txt, means.txt, the filled 'weights' folder, the 'compr_data' folder
and the 'decompr_data' destination folder are needed.

The main calls the data_decompressor('compr_data.txt', 2048, img=1) method and saves
the results to 'decompr_data/decompr_data.txt'.

.. py:function:: data_decompressor(fname, n_bins, img=0)

    Takes a fname txt file, containing an array of 20 elements numpy arrays (shape (,20)), and applies the inverse
    of the lossy compression of :ref:`data_compressor <compress>`.

    :param fname: name of the .txt source file
    :type fname: str
    :param n_bits: number of the bins used for the compression
    :type n_bits: int
    :param img: if it is 1, images of the distribution of the data at the various steps (before the decompression,
    after converting to gaussians, after reconverting to the original distributions) are produced; defaults to 0
    :type img: int
    :raise OSError: if file does not exist or is not in .txt format
    :return: the data contained in fname, decompressed
    :rtype: numpy.array (,20)