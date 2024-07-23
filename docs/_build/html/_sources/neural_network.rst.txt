neural_network.py
=================

The RealVPN machine learning algorithm used to convert the data of symmetric 5x5 matrices into 15 normal distributions.

.. py:function:: coupling(input_shape)

    The coupling layer used by the neural network. It has an initial normalization layer, then it splits into two
    branches which return the parameters s and t used for the normalizing flow.

.. py:class:: RealNVP(keras.Model)

    RealNVP finds the change of coordinates from the original data to normal distributions.
    It cascades invertible transformations depending on the parameters s and t.
    The loss function is the sum of the log likelihood of the normal distribution
    and the log determinant of the jacobian of the transformation.

There main has the options to save the training weight in the weight folder and load them for future training,
or even just load them without further training.
This options can be selected by modifying the bools SaveWeights, LoadWeights and TrainModel.
At the end a plot containing the final transformed data is presented.

