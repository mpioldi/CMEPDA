# CMEPDA
This is Matteo Pioldi and Lorenzo Pierro's project for the course “Computing Methods for Experimental Physics and Data Analysis”. It  consists of an algorithm for lossy data compression based on a Normalizing Flow Neural Network. The idea is based on the observation that, thanks to the Normalizing Flow technique, arbitrary distributions can be bijectively mapped into gaussian ones. Then, dividing a gaussian into bins of equal total probability is an easy task. Each bin can be labelled with an integer, so that each value of the original dataset is associated with the label of the destination bin. Storing data in the forms of such integer labels means losing information, but it also allows to reduce the space required to store them.

We show an example implementation working on the 5D curvilinear covariance matrix of CMS opendata dataset https://opendata.cern.ch/record/7722 . The goal is to obtain a parametric compression function that takes as an input not only the 15 unique elements of the covariance matrix to be compressed, but also the parameters (qoverp, lambda, phi, dxy, dsz). The proposed way to achieve lossy compression is to divide into bins of equal volume data with a known distribution, to be obtained from the original ones by a change of coordinates. The task of finding the adequate change of coordinates can be executed with a neural network. In this specific case, we chose the RealNVP (Real Non-Volume-Preserving) framework. This method allows to find a change of coordinates, as explained in the following.

The project includes a neural network, coded in neural_network.py, that can be trained on the chosen dataset and store the data needed to reconstruct a change of coordinates that brings the data distribution into a gaussian. The change of coordinates is then employed by the compressor.py and decompressor.py scripts. In them, data is compressed by passing it through the change of coordinates, mapping to a gaussian and dividing it into equal bins and decompressed by taking the inverse operations.

In addition, traducer methods to convert data from the ROOT format to a .txt file and vice versa have been written.

Finally, scripts to evaluate the performance of the compressor are present: compare.py and eigen.py, respectively considering the relative error committed by compressing and decompressing the data, and compression_comparator.py, that instead plots side-to-side the performance of the Normalizing Flow compressor with a compression by cutting the least significative digits of the mantissa.

The scripts are stored in the 'code' folder, while html documentation is in the 'docs' one.
