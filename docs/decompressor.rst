decompressor.py
=====

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
    :raise OSError: if file does not exist or is not in .txt format
    :return: the data contained in fname, decompressed
    :rtype: numpy.array (,20)