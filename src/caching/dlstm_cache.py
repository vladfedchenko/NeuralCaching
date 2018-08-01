"""
This module contains the implementation of the caching policy described in "Content Caching with Deep Long Short-Term
Memory Network" by Haitian Pang, Jiangchuan Liu,Fellow, IEEE, Shizhi Tang, Ruixiao Zhang, and Lifeng Sun,Member, IEEE.
"""
from neural_nets import LSTMSoftmax
from caching.abstract_cache import AbstractCache
import numpy as np
import torch


class DLSTMCache(AbstractCache):

    # region Private variables

    __input_len = 0
    __out_len = 0

    __lstm_net = None
    __learning_rate = 0.0

    __training_lag = 0
    __cur_lag = 0

    __alpha = 0.0

    __request_log = None
    __max_request_log_size = 0

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self,
                 size: int,
                 input_len: int,
                 out_len: int,
                 lstm_layers: int = 2,
                 cell_number: int = 16,
                 training_lag: int = 5,
                 alpha: float = 0.125,
                 learning_rate: float = 0.01,
                 max_request_log_size: int = 10000):
        """
        Create new DLSTMCache object. Described in "Content Caching with Deep Long Short-Term
        Memory Network" by Haitian Pang, Jiangchuan Liu,Fellow, IEEE, Shizhi Tang, Ruixiao Zhang,
        and Lifeng Sun,Member, IEEE.
        :param size: Size of the cache.
        :param input_len: Length of input for LSTM network.
        :param out_len: Length of output (equals number of objects in trace)
        :param lstm_layers: Layers in LSTM network.
        :param cell_number: Number of cells in LSTM network.
        :param training_lag: Number of cache misses before training is initialized.
        :param alpha: Parameter to determine true caching priority for training.
        :param learning_rate: Learning rate for training.
        :param max_request_log_size: How many last requests to store for training.
        """
        super().__init__(size)

        self.__input_len = input_len
        self.__out_len = out_len

        self.__lstm_net = LSTMSoftmax([input_len] + ([cell_number] * lstm_layers) + [out_len])
        # self.__lstm_net = LSTMSoftmax([input_len, cell_number, out_len])
        self.__learning_rate = learning_rate

        self.__training_lag = training_lag
        self.__alpha = alpha

        self.__request_log = []
        self.__max_request_log_size = max_request_log_size

    # endregion

    # region Private methods

    def __item_priority(self, index):
        ret = 1.0 - ((index + 1.0) / self.__out_len)**self.__alpha
        return ret

    def __calc_priority(self, req_sequence):
        priority = [0.0] * self.__out_len
        for i, item in enumerate(req_sequence):
            priority[item] += self.__item_priority(i)

        priority = np.exp(priority)
        priority /= np.sum(priority, axis=0)

        return priority

    def __log_request(self, id_):
        self.__request_log.append(id_)
        if len(self.__request_log) > self.__max_request_log_size:
            del self.__request_log[0]

    def __train_online(self):
        if len(self.__request_log) >= self.__input_len + self.__out_len:
            i_start = self.__input_len + self.__out_len

            inp_list = []
            outp_list = []
            for i in range(i_start, len(self.__request_log) + 1):
                j = i - self.__input_len - self.__out_len
                inp_list.append(self.__request_log[j:j+self.__input_len])
                outp_list.append(self.__calc_priority(self.__request_log[j+self.__input_len:i]))

            inp = torch.Tensor(inp_list)
            outp = torch.Tensor(outp_list)

            self.__lstm_net.backpropagation_learn(inp, outp, learn_rate=self.__learning_rate, stochastic=True)

    def __decide_replace(self, id_, size, pred: torch.Tensor):
        pred = pred.reshape(1, self.__out_len)
        min_pred = 10
        min_index = 0

        cached = self.cached_objects()
        for c in cached:
            if pred[0, c] < min_pred:
                min_pred = pred[0, c]
                min_index = c

        if pred[0, id_] > min_pred:
            self._remove_object(min_index)
            self._store_object(id_, size)

    # endregion

    # region Protected methods

    def _process_cache_hit(self, id_, size, time, metadata):
        self.__log_request(id_)

    def _process_cache_miss(self, id_, size, time, metadata):
        self.__log_request(id_)
        self.__cur_lag += 1
        if self.__cur_lag == self.__training_lag:
            self.__cur_lag = 0
            self.__train_online()

        if self._free_cache < size:
            inp = torch.Tensor([self.__request_log[-5:]])

            pred = self.__lstm_net(inp)

            self.__decide_replace(id_, size, pred)
        else:
            self._store_object(id_, size)

    # endregion

    # region Public methods

    # endregion
