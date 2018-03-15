"""
This module contains the implementation of simple forward propagation neural network layer.
"""
import numpy as np
import math
from types import *


class NeuralNetLayer:
    """
    NeuralNetLayer is the implementation of a simple forward propagation neural network layer.
    """
    # region Private variables

    __coef_matrix = None
    __activation = None

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self, neurons: int, prev_layer_neurons: int, activation_func: FunctionType=math.tanh):
        """
        Construct a new NeuralNetLayer object.
        :param neurons: Number of neurons in current layer.
        :param prev_layer_neurons: Number of neurons in previous layer.
        :param activation_func: Neuron activation function. Pass None if no activation required.
        """
        self.__coef_matrix = np.random.rand(prev_layer_neurons, neurons)
        if activation_func is not None:
            self.__activation = np.vectorize(activation_func)
        else:
            self.__activation = None

    # endregion

    # region Private methods

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    def propagate(self, inputs: np.ndarray):
        """
        Takes a row-vector of layer inputs and produces a row-vector of layer outputs.
        :param inputs: Row-vector of inputs.
        :return: npumpy.ndarray -> Row-vector of outputs.
        """
        assert (inputs.shape[1] == self.__coef_matrix.shape[0])
        outs = np.matmul(inputs, self.__coef_matrix)
        if self.__activation is not None:
            outs = self.__activation(outs)
        return outs

    # endregion