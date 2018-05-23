"""
This module contains implementation of a feedforward NN using pytorch library.
"""
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as functional
import numpy as np
from tqdm import tqdm
import sys
import math


class TorchFeedforwardNN(nn.Module):

    # region Private variables

    # __fc_hidden = None  # input + hidden layers list
    # __fc_output = None  # output layer

    __hidden_activation = False
    __out_activation = False

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self,
                 neurons_list: List[int],
                 hidden_activation: str=None,
                 out_activation: str=None):
        """
        FeedforwardNeuralNet a new NeuralNetLayer object.
        :param neurons_list: List of neurons counts in layers of NN.
        :param hidden_activation: Activation to use on hidden layers.
        :param out_activation: Activation to use on out layer.
        """
        super().__init__()

        self.__fc_hidden = []
        for i in range(1, len(neurons_list) - 1):
            layer = nn.Linear(neurons_list[i - 1], neurons_list[i]).double()

            self.__fc_hidden.append(layer)
            self.add_module("__fc{0}".format(i), layer)

        self.__criterion = nn.MSELoss()

        layer = nn.Linear(neurons_list[-2], neurons_list[-1]).double()
        self.__fc_output = layer

        self.__hidden_activation = hidden_activation
        self.__out_activation = out_activation

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

        out = x
        for layer in self.__fc_hidden:
            out = layer(out)

            if self.__hidden_activation == "sigmoid":
                out = functional.sigmoid(out)
            elif self.__hidden_activation == "relu":
                out = functional.relu(out)
            elif self.__hidden_activation == "l_relu":
                out = functional.leaky_relu(out)

        out = self.__fc_output(out)

        if self.__out_activation == "sigmoid":
            out = functional.sigmoid(out)
        elif self.__out_activation == "relu":
            out = functional.relu(out)
        elif self.__out_activation == "l_relu":
            out = functional.leaky_relu(out)

        return out

    def backpropagation_learn(self, inputs: torch.Tensor,
                              outputs: torch.Tensor,
                              learn_rate: float = 0.1,
                              stochastic: bool = True,
                              show_progress: bool = False):
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

        if stochastic:
            if show_progress:
                iterable = tqdm(range(n), desc="Samples processed", unit="samples")
            else:
                iterable = range(n)
            for sample in iterable:
                inp = inputs[sample, :]
                target = outputs[sample, :]

                self.zero_grad()
                out = self(inp)

                loss = self.__criterion(out, target)
                loss.backward()
                for f in self.parameters():
                    f.data.sub_(f.grad.data * learn_rate)
        else:
            self.zero_grad()
            out = self(inputs)

            loss = self.__criterion(out, outputs)
            loss.backward()
            for f in self.parameters():
                f.data.sub_(f.grad.data * learn_rate)

    def evaluate(self,
                 inputs: torch.Tensor,
                 outputs: torch.Tensor) -> float:
        """
        Evaluate the NN by calculating mean error using provided error function.
        :param inputs: Matrix of inputs.
        :param outputs: Matrix of outputs.
        :return: Error using provided on creation error function.
        """
        n = inputs.shape[0]
        calc_outp = self(inputs).detach()

        self.zero_grad()
        loss = self.__criterion(outputs, calc_outp)
        if loss.device.type.startswith("cuda"):
            loss = loss.cpu()
        loss_np = loss.numpy()

        return float(loss_np)

    # endregion
