"""
This module contains the implementation of cache policy using feedforward NN.
"""
from caching.abstract_cache import AbstractCache
from neural_nets import TorchFeedforwardNN
from helpers.collections import FullCounter, PriorityDict
import random
import numpy as np
import torch


class FeedforwardNNCacheFullTorch(AbstractCache):

    # region Private variables

    __counters = None
    __trained_net = None
    __time_window = 0.0
    __processed_windows = 0
    __from_window_start = 0.0
    __priority_dict = None
    __update_sample_size = 0

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self,
                 size: int,
                 trained_net: TorchFeedforwardNN,
                 counter_num: int,
                 time_window: float,
                 update_sample_size: int=5):
        """
        Construct a new FeedforwardNNCache object.
        :param size: Size of cache.
        :param trained_net: Trained neural network.
        :param counter_num: Number of counters.
        :param time_window: Time window of one sketch activity.
        :param update_sample_size: How many items popularity to update when processing cache hit.
        """
        super().__init__(size)
        self.__trained_net = trained_net

        self.__counters = []
        for i in range(counter_num):
            sketch = FullCounter()
            self.__counters.append(sketch)

        self.__time_window = time_window
        self.__processed_windows = 0
        self.__from_window_start = 0.0

        self.__priority_dict = PriorityDict()

        self.__update_sample_size = update_sample_size

    # endregion

    # region Private methods

    def __predict_pop(self, id_, time: float) -> float:
        """
        Predict popularity of object using NN and sketches.
        :param id_: ID of the object.
        :param time: Time of arrival.
        :return: Predicted popularity.
        """
        prediction_row = []
        for sketch in self.__counters:
            frac = sketch.get_request_fraction(id_)
            frac = -np.log(frac + 10**-15)
            prediction_row.append(frac)

        window_time = (time - self.__time_window * self.__processed_windows) / self.__time_window
        prediction_row.append(window_time)

        matr = torch.from_numpy(np.matrix([prediction_row]))
        pop_log = float(self.__trained_net(matr))
        pop = np.exp(-pop_log) - 10**-15
        return pop

    def __update_time(self, time: float):
        """
        Updates time related activity - active sketches, time from window start, etc.
        :param time: Time of object arrival.
        """
        added_time = time - self.__time_window * self.__processed_windows
        self.__from_window_start += added_time

        while self.__from_window_start > self.__time_window:
            self.__processed_windows += 1
            self.__from_window_start -= self.__time_window
            del self.__counters[0]
            sketch = FullCounter()
            self.__counters.append(sketch)

    # endregion

    # region Protected methods

    def _process_cache_hit(self, id_, size, time):
        self.__update_time(time)
        self.__counters[-1].update_counters(id_)
        if len(self.__priority_dict) < self.__update_sample_size:
            real_update_size = len(self.__priority_dict)
        else:
            real_update_size = self.__update_sample_size

        sample = random.sample(self.__priority_dict.keys(), real_update_size)
        for id_ in sample:
            pred_pop = self.__predict_pop(id_, time)
            self.__priority_dict[id_] = pred_pop

    def _process_cache_miss(self, id_, size, time):
        self.__update_time(time)
        self.__counters[-1].update_counters(id_)
        pred_pop = self.__predict_pop(id_, time)
        if self._free_cache > 0:
            self._store_object(id_, size)
            self.__priority_dict[id_] = pred_pop

        else:
            candidate = self.__priority_dict.smallest()
            if pred_pop > self.__priority_dict[candidate]:
                self._remove_object(candidate)
                self._store_object(id_, size)
                self.__priority_dict.pop_smallest()
                self.__priority_dict[id_] = pred_pop

    # endregion

    # region Public methods

    # endregion
