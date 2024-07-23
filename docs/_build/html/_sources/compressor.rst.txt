.. _compressor:

compressor.py
=====

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
    :raise OSError: if file does not exist or is not in .txt format
    :return: the data contained in fname, compressed
    :rtype: numpy.array (,20)