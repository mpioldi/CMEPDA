Usage
=====

Requirements
-----------------

To use NormFlow compression, you will need Python 3, ROOT and the following Python packages:

.. code-block:: console

    import os #only needed to run the tests
    import numpy
    import ROOT 
    import keras


Translating from ROOT to numpy an uncompressed database
--------------------------------------------------------

.. py:function:: RootToNumpy(name, treename)

    Take a name.root file, containing a 'treename' tree of the expected shape (a 'Vector'
    branch made of 5 elements arrays of doubles and a 'Matrix' branch of 15 elements arrays
    of doubles) and converts it to 20 elements numpy arrays (shape (,20)).

    :param name: name of the .root file (without extension)
    :type name: str
    :param treename: name of the tree inside the .root file
    :type treename: str
    :return: the data contained in the tree 'treename', in as a numpy array
    :rtype: numpy.array (,20)

