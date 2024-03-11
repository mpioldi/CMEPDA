Usage
=====

Requirements
-----------------

To use NormFlow compression, you will need Python 3, ROOT and the following Python packages:

.. code-block:: console

    import os #only needed to run the tests
    import numpy
    import ROOT 
    import tensorflow
    import tensorflow_probability
    import matplotlib.pyplot



Translating from ROOT to numpy an uncompressed database
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