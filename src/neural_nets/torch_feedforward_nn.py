"""
This module contains implementation of a feedforward NN using pytorch library.
"""
import torch
import torch.nn as nn
from typing import List


class TorchFeedforwardNN(nn.Module):

    # region Private variables

    __fc_hidden = None  # input + hidden layers list
    __fc_output = None  # output layer

    __use_sigmoid_hidden = False
    __use_sigmoid_out = False

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self,
                 neurons_list: List[int],
                 use_sigmoid_hidden: bool=False,
                 use_sigmoid_out: bool=False):
        """
        FeedforwardNeuralNet a new NeuralNetLayer object.
        :param neurons_list: List of neurons counts in layers of NN.
        :param use_sigmoid_hidden: Pass True if you want to apply sigmoid at hidden layers.
        :param use_sigmoid_out: Pass True if you want to apply sigmoid at output layer.
        """
        super().__init__()

        self.__fc_hidden = []
        self.__hidden_sigmoids = []
        for i in range(1, len(neurons_list) - 1):
            layer = nn.Linear(neurons_list[i - 1], neurons_list[i])

            self.__fc_hidden.append(layer)
            self.add_module("fc{0}".format(i), layer)

        self.__use_sigmoid_hidden = use_sigmoid_hidden
        self.__use_sigmoid_out = use_sigmoid_out

        self.__fc_output = nn.Linear(neurons_list[-2], neurons_list[-1])

    # endregion

    # region Private methods

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    def forward(self, x):
        """
        Feedforward pass of the input.
        :param x: Input.
        :return: Feedforwarded output.
        """
        sigm = nn.Sigmoid()

        out = x
        for layer in self.__fc_hidden:
            out = layer(out)
            if self.__use_sigmoid_hidden:
                out = sigm(out)

        out = self.__fc_output(out)
        if self.__use_sigmoid_out:
            out = sigm(out)

        return out

    # endregion
