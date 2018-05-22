"""
This module contains the implementation of simple feedforward neural network.
"""
from neural_nets.neuron_layer import *
from typing import Callable, TypeVar
from typing import List
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


def kl_divergence(y, calc_y):
    return -y * np.log(calc_y / y)


def kl_divergence_der(y, calc_y):
    return -y / calc_y


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
                 hidden_activation: str = None,
                 out_activation: str = None,
                 error_func: str = None):
        """
        FeedforwardNeuralNet a new NeuralNetLayer object.
        :param neurons_list: List of neurons counts in layers of NN.
        :param hidden_activation: Activation to use on hidden layers.
        :param out_activation: Activation to use on out layer.
        :param error_func: Error function to use.
        :param error_deriv: Derivative of the error function.
        """
        self.__layers = []
        for i in range(1, len(neurons_list)):
            prev_layer = neurons_list[i - 1]
            cur_layer = neurons_list[i]

            out_activ = lin
            out_activ_deriv = lin_deriv

            if out_activation == "sigmoid":
                out_activ = sigmoid
                out_activ_deriv = sigmoid_deriv
            elif out_activation == "relu":
                out_activ = relu
                out_activ_deriv = relu_deriv
            elif out_activation == "l_relu":
                out_activ = l_relu
                out_activ_deriv = l_relu_deriv

            hidden_activ = lin
            hidden_activ_deriv = lin_deriv

            if hidden_activation == "sigmoid":
                hidden_activ = sigmoid
                hidden_activ_deriv = sigmoid_deriv
            elif hidden_activation == "relu":
                hidden_activ = relu
                hidden_activ_deriv = relu_deriv
            elif hidden_activation == "l_relu":
                hidden_activ = l_relu
                hidden_activ_deriv = l_relu_deriv

            if i == len(neurons_list) - 1:
                layer = NeuralNetLayer(cur_layer, prev_layer, out_activ, out_activ_deriv)
            else:
                layer = NeuralNetLayer(cur_layer, prev_layer, hidden_activ, hidden_activ_deriv)
            self.__layers.append(layer)

        error_func_f = squared_error
        error_deriv = squared_error_der

        if error_func == "mse":
            error_func_f = squared_error
            error_deriv = squared_error_der
        elif error_func == "kl":
            error_func_f = kl_divergence
            error_deriv = kl_divergence_der

        self.__error_func = error_func_f
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

    def get_weights(self) -> List[np.matrix]:
        """
        Get weights of the NN.
        :return: List of matrices which are weights of the layers of the NN.
        """
        to_ret = []
        for layer in self.__layers:
            to_ret.append(layer.get_coefficients())
        return to_ret

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
