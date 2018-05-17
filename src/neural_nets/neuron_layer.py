"""
This module contains the implementation of simple feedforward neural network layer.
"""
import numpy as np
import math
from types import FunctionType


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


def lin(x):
    return x


def lin_deriv(_):
    return 1


class NeuralNetLayer:
    """
    NeuralNetLayer is the implementation of a simple feedforward neural network layer.
    """
    # region Private variables

    __coef_matrix = None
    __activation = None
    __activation_deriv = None

    __inputs_mem = None
    __outputs_mem = None

    __delta_weights = None
    __samples_num = 0

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self, neurons: int,
                 prev_layer_neurons: int,
                 activation_func: FunctionType=sigmoid,
                 activation_deriv: FunctionType=sigmoid_deriv):
        """
        Construct a new NeuralNetLayer object.
        :param neurons: Number of neurons in current layer.
        :param prev_layer_neurons: Number of neurons in previous layer.
        :param activation_func: Neuron activation function. Pass None if no activation required.
        """
        self.__coef_matrix = np.random.rand(prev_layer_neurons + 1, neurons, dtype=np.float256) - 0.5

        if activation_func is not None and activation_deriv is not None:
            self.__activation = np.vectorize(activation_func)
            self.__activation_deriv = np.vectorize(activation_deriv)
        else:
            self.__activation = None
            self.__activation_deriv = np.vectorize(lin_deriv)

    # endregion

    # region Private methods

    def __reset_memory(self):
        """
        Forget previous inputs and outputs
        """
        self.__inputs_mem = None
        self.__outputs_mem = None

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    def get_coefficients(self) -> np.matrix:
        """
        Get copy of this layer coefficients.
        :return: Matrix of coefficients.
        """
        return self.__coef_matrix.copy()

    def propagate(self, inputs: np.matrix):
        """
        Takes a column-vector of layer inputs and produces a column-vector of layer outputs.
        :param inputs: Row-vector of inputs.
        :return: numpy.matrix -> Row-vector of outputs.
        """
        assert (inputs.shape[0] + 1 == self.__coef_matrix.shape[0])
        self.__reset_memory()
        inputs = np.concatenate(([[1.0]], inputs), axis=0)

        outs = np.matmul(inputs.T, self.__coef_matrix)
        if self.__activation is not None:
            outs = self.__activation(outs)
        return outs.T

    def propagate_with_mem(self, inputs: np.matrix):
        """
        Propagation with memorization of inputs and outputs.
        :param inputs: Row-vector of inputs.
        :return: numpy.matrix -> Row-vector of outputs.
        """
        self.__outputs_mem = self.propagate(inputs)
        self.__inputs_mem = np.concatenate(([[1.0]], inputs), axis=0)
        return self.__outputs_mem

    def backpropagate_stoch(self, error: np.matrix, learn_rate: float):
        """
        Backpropagation of error and weight update for current layer.
        :param error: Errors from next layer.
        :param learn_rate: Learning rate.
        :return: np.matrix -> Errors to backpropagate to previous layer.
        """
        deltas = np.multiply(error, self.__activation_deriv(self.__outputs_mem))
        err_to_pass = (self.__coef_matrix * deltas)[1:, :]

        delta_weight = self.__inputs_mem * deltas.T * learn_rate
        self.__coef_matrix -= delta_weight

        return err_to_pass

    def update_weights(self):
        """
        Run this function after running backpropagate_batch on every sample of the batch
        """
        self.__coef_matrix -= self.__delta_weights / self.__samples_num
        self.__delta_weights = None
        self.__samples_num = 0

    def backpropagate_batch(self, error: np.matrix, learn_rate: float):
        """
        Backpropagation of error. To update weights execute update_weights() function.
        :param error: Errors from next layer.
        :param learn_rate: Learning rate.
        :return: np.matrix -> Errors to backpropagate to previous layer.
        """
        deltas = np.multiply(error, self.__activation_deriv(self.__outputs_mem))
        err_to_pass = (self.__coef_matrix * deltas)[1:, :]

        delta_weight = self.__inputs_mem * deltas.T * learn_rate

        if self.__samples_num == 0:
            self.__delta_weights = delta_weight
        else:
            self.__delta_weights += delta_weight
        self.__samples_num += 1

        return err_to_pass

    # endregion
