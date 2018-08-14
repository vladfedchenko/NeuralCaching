"""
This module containts the implementation of LSTM NN using pytorch library.
"""
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
from tqdm import tqdm


class LSTMSoftmax(torch.nn.Module):

    # region Private variables

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self,
                 neurons_list: List[int]):
        """
        Construct a new LSTMSoftmax object.
        :param neurons_list: List of neurons counts in layers of NN.
        """
        super().__init__()

        self.__lstm_hidden = []
        self.__lstm_states = []
        for i in range(1, len(neurons_list) - 1):
            layer = nn.LSTM(neurons_list[i - 1], neurons_list[i], 2)

            self.__lstm_hidden.append(layer)
            self.add_module("__lstm{0}".format(i), layer)

            h0 = torch.randn(2, 1, neurons_list[i]).detach()
            c0 = torch.randn(2, 1, neurons_list[i]).detach()

            self.__lstm_states.append((h0, c0))

        layer = nn.Linear(neurons_list[-2], neurons_list[-1])
        self.__fc_output = layer

        self.__criterion = torch.nn.BCELoss()

    # endregion

    # region Private methods

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    def forward(self, x: torch.Tensor):
        """
        Forward input through the network.
        :param x: Input. The shape should be [batch_size, input_size].
        :return: Output. The shape is [batch_size, output_size].
        """
        batch_size = x.shape[0]
        inp_size = x.shape[1]

        final_out = None
        for i in range(batch_size):

            cur_inp = x[i:i+1, :].reshape((1, 1, inp_size))

            out = cur_inp
            new_states = []
            for layer, hid_state in zip(self.__lstm_hidden, self.__lstm_states):
                out, (h1, c1) = layer(out, hid_state)
                new_states.append((h1.detach(), c1.detach()))

            self.__lstm_states = new_states

            out = self.__fc_output(out.reshape(1, out.shape[2]))
            out = F.softmax(out, dim=1)

            if final_out is None:
                final_out = out
            else:
                final_out = torch.cat((final_out, out), 0)

        return final_out

    def backpropagation_learn(self, inputs: torch.Tensor,
                              outputs: torch.Tensor,
                              learn_rate: float = 0.1,
                              stochastic: bool = True,
                              show_progress: bool = False,
                              weight: float = 1.0):
        """
        Update the weights of the NN using error backward propagation algorithm.
        Stochastic gradient descent is the basis.
        :param inputs: Matrix of inputs.
        :param outputs: Matrix of outputs.
        :param learn_rate: Learning rate parameter.
        :param stochastic: Use stochastic gradient descent.
        :param show_progress: Show progress bar.
        :param weight: Apply weight to loss.
        """
        n = inputs.shape[0]

        if stochastic:
            if show_progress:
                iterable = tqdm(range(n), desc="Samples processed", unit="samples")
            else:
                iterable = range(n)
            for sample in iterable:
                inp = inputs[sample:sample+1, :]
                target = outputs[sample:sample+1, :]

                self.zero_grad()
                out = self(inp)

                loss = self.__criterion(out, target) * weight
                loss.backward()
                for f in self.parameters():
                    f.data.sub_(f.grad.data * learn_rate)
        else:
            self.zero_grad()
            out = self(inputs)

            loss = self.__criterion(out, outputs) * weight
            loss.backward(retain_graph=True)
            for f in self.parameters():
                f.data.sub_(f.grad.data * learn_rate)

    def evaluate(self,
                 inputs: torch.Tensor,
                 outputs: torch.Tensor) -> float:
        """
        Evaluate the NN by calculating error using provided error function.
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
