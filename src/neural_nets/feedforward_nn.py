"""
This module contains the implementation of simple feedforward neural network.
"""
from neural_nets.neuron_layer import NeuralNetLayer, sigmoid, sigmoid_deriv
from types import FunctionType
import numpy as np


def squared_error(y, calc_y):
    tmp = y - calc_y
    return 1.0 / 2.0 * np.multiply(tmp, tmp)


def squared_error_der(y, calc_y):
    return calc_y - y


class FeedforwardNeuralNet:
    """
    FeedforwardNeuralNet is the implementation of a simple feedforward neural network.
    """
    # region Private variables

    __layers = None
    __error_func = None
    __error_deriv = None

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self,
                 neurons_list: list,
                 internal_activ: FunctionType=sigmoid,
                 internal_activ_deriv: FunctionType=sigmoid_deriv,
                 out_activ: FunctionType=None,
                 out_activ_deriv: FunctionType=None,
                 error_func: FunctionType = squared_error,
                 error_deriv: FunctionType=squared_error_der):
        """
        FeedforwardNeuralNet a new NeuralNetLayer object.
        :param neurons_list: List of neurons counts in layers of NN.
        :param internal_activ: Activation function for internal layers.
        :param internal_activ_deriv: Derivative of activation function for internal layers.
        :param out_activ: Activation function for output layer.
        :param out_activ_deriv: Derivative of activation function for output layer.
        :param error_deriv: Derivative of the error function.
        """
        self.__layers = []
        for i in range(1, len(neurons_list)):
            prev_layer = neurons_list[i - 1]
            cur_layer = neurons_list[i]

            if i == len(neurons_list) - 1:
                layer = NeuralNetLayer(cur_layer, prev_layer, out_activ, out_activ_deriv)
            else:
                layer = NeuralNetLayer(cur_layer, prev_layer, internal_activ, internal_activ_deriv)
            self.__layers.append(layer)

        self.__error_func = error_func
        self.__error_deriv = error_deriv

    # endregion

    # region Private methods

    def __feedforward_with_mem(self, inputs: np.matrix):
        """
        Same as feedforward, but layers memorize inputs and outputs.
        :param inputs: Row-vector of inputs.
        :return: numpy.matrix -> Row-vector of outputs.
        """
        outs = inputs
        for layer in self.__layers:
            outs = layer.propagate_with_mem(outs)
        return outs

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    def feedforward(self, inputs: np.matrix):
        """
        Takes a row-vector of input layer inputs and produces a row-vector of output layer outputs.
        :param inputs: Row-vector of inputs.
        :return: numpy.matrix -> Row-vector of outputs.
        """
        outs = inputs
        for layer in self.__layers:
            outs = layer.propagate(outs)
        return outs

    def backpropagation_learn(self, inputs: np.matrix, outputs: np.matrix, iterations: int=100, learn_rate: float=0.1):
        """
        Update the weights of the NN using error backward propagation algorithm.
        Stochastic gradient descent is the basis.
        :param inputs: Matrix of inputs.
        :param outputs: Matrix of outputs.
        :param iterations: Number of iterations to learn.
        :param learn_rate: Learning rate parameter.
        """
        n = inputs.shape[0]
        for i in range(iterations):
            for sample in range(n):
                inp = inputs[sample, :].T
                outp = outputs[sample, :].T
                calc_outp = self.__feedforward_with_mem(inp)

                error = self.__error_deriv(outp, calc_outp)
                for layer in reversed(self.__layers):
                    error = layer.backpropagate(error, learn_rate)

    def evaluate(self, inputs: np.matrix, outputs: np.matrix) -> float:
        """
        Evaluate the NN by calculating mean error using provided error function.
        :param inputs: Matrix of inputs.
        :param outputs: Matrix of outputs.
        :return: Average error using provided on creation error function.
        """
        n = inputs.shape[0]
        full_err = 0.0
        for sample in range(n):
            inp = inputs[sample, :].T
            outp = outputs[sample, :].T
            calc_outp = self.feedforward(inp)
            err = self.__error_func(outp, calc_outp)
            full_err += np.sum(err)

        return full_err / n

    # endregion
