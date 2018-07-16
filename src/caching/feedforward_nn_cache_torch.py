"""
This module contains the implementation of cache policy using feedforward NN.
"""
from caching.abstract_cache import AbstractCache
from neural_nets import TorchFeedforwardNN
from helpers.collections import CountMinSketch, PriorityDict
import random
import numpy as np
import torch


class FeedforwardNNCacheTorch(AbstractCache):

    # region Private variables

    __count_min_sketches = None
    __additive_factor = 0.0
    __probability = 0.0
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
                 sketches_num: int,
                 sketches_additive_factor: float,
                 sketches_probability: float,
                 time_window: float,
                 update_sample_size: int=5):
        """
        Construct a new FeedforwardNNCache object.
        :param size: Size of cache.
        :param trained_net: Trained neural network.
        :param sketches_num: Number of count-min sketches.
        :param sketches_additive_factor: Additive factor of sketches.
        :param sketches_probability: Probability for sketches.
        :param time_window: Time window of one sketch activity.
        :param update_sample_size: How many items popularity to update when processing cache hit.
        """
        super().__init__(size)
        self.__trained_net = trained_net

        self.__count_min_sketches = []
        self.__additive_factor = sketches_additive_factor
        self.__probability = sketches_probability
        for i in range(sketches_num):
            sketch = CountMinSketch.construct_by_constraints(sketches_additive_factor, sketches_probability)
            self.__count_min_sketches.append(sketch)

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
        for sketch in self.__count_min_sketches:
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
            del self.__count_min_sketches[0]
            sketch = CountMinSketch.construct_by_constraints(self.__additive_factor, self.__probability)
            self.__count_min_sketches.append(sketch)

    # endregion

    # region Protected methods

    def _process_cache_hit(self, id_, size, time, metadata):
        self.__update_time(time)
        self.__count_min_sketches[-1].update_counters(id_)
        if len(self.__priority_dict) < self.__update_sample_size:
            real_update_size = len(self.__priority_dict)
        else:
            real_update_size = self.__update_sample_size

        pred_pop = self.__predict_pop(id_, time)
        self.__priority_dict[id_] = pred_pop

        sample = random.sample(self.__priority_dict.keys(), real_update_size)
        for i in sample:
            pred_pop = self.__predict_pop(i, time)
            self.__priority_dict[i] = pred_pop

    def _process_cache_miss(self, id_, size, time, metadata):
        self.__update_time(time)
        self.__count_min_sketches[-1].update_counters(id_)
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
