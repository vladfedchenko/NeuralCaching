"""
This module contains the implementation of simple feedforward neural network.
"""
from neural_nets.neuron_layer import NeuralNetLayer, sigmoid, sigmoid_deriv
from typing import Callable, TypeVar
import numpy as np
from tqdm import tqdm
import sys
import math


ActivationFuncType = TypeVar("ActivationFuncType", None, Callable[[float], float])


def squared_error(y, calc_y):
    tmp = y - calc_y
    return 1.0 / 2.0 * np.multiply(tmp, tmp)


def squared_error_der(y, calc_y):
    return calc_y - y


def poisson_error_der(y, calc_y):
    return -1.0 + y / calc_y


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
                 internal_activ: ActivationFuncType=sigmoid,
                 internal_activ_deriv: ActivationFuncType=sigmoid_deriv,
                 out_activ: ActivationFuncType=None,
                 out_activ_deriv: ActivationFuncType=None,
                 error_func: Callable[[float, float], float]=squared_error,
                 error_deriv: Callable[[float, float], float]=squared_error_der):
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

    def backpropagation_learn(self, inputs: np.matrix,
                              outputs: np.matrix,
                              learn_rate: float=0.1,
                              stochastic: bool=True,
                              show_progress: bool=False):
        """
        Update the weights of the NN using error backward propagation algorithm.
        Stochastic gradient descent is the basis.
        :param inputs: Matrix of inputs.
        :param outputs: Matrix of outputs.
        :param learn_rate: Learning rate parameter.
        :param stochastic: Use stochastic gradient descent.
        :param show_progress: Show progress bar.
        """
        n = inputs.shape[0]
        if show_progress:
            iterable = tqdm(range(n), desc="Samples processed", unit="samples")
        else:
            iterable = range(n)
        for sample in iterable:
            inp = inputs[sample, :].T
            outp = outputs[sample, :].T
            calc_outp = self.__feedforward_with_mem(inp)

            error = self.__error_deriv(outp, calc_outp)
            for layer in reversed(self.__layers):
                if stochastic:
                    error = layer.backpropagate_stoch(error, learn_rate)
                else:
                    error = layer.backpropagate_batch(error, learn_rate)

        if not stochastic:
            for layer in reversed(self.__layers):
                layer.update_weights()

    def evaluate(self,
                 inputs: np.matrix,
                 outputs: np.matrix,
                 show_progress: bool=False) -> (float, float, float):
        """
        Evaluate the NN by calculating mean error using provided error function.
        :param inputs: Matrix of inputs.
        :param outputs: Matrix of outputs.
        :param show_progress: Show progress bar.
        :return: Mean, standard deviation, min, max errors using provided on creation error function.
        """
        n = inputs.shape[0]
        full_err = 0.0
        min_error = sys.float_info.max
        max_error = 0.0
        if show_progress:
            iterable = tqdm(range(n), desc="Evaluated samples", unit="samples")
        else:
            iterable = range(n)

        errors = []
        for sample in iterable:
            inp = inputs[sample, :].T
            outp = outputs[sample, :].T
            calc_outp = self.feedforward(inp)
            err = np.sum(self.__error_func(outp, calc_outp))

            if err < min_error:
                min_error = err

            if err > max_error:
                max_error = err

            errors.append(err)

            full_err += err

        mean = full_err / n
        deviation = math.sqrt(np.sum([(x - mean)**2 for x in errors]) / n)

        return mean, deviation, min_error, max_error

    # endregion
