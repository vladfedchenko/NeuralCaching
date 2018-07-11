"""
This caching policy mimics neural-network based policy. But instead of predicting the popularity by a NN it takes an
average of previously observed popularity.
"""
from caching.abstract_cache import AbstractCache
from helpers.collections import FullCounter, PriorityDict
import random
import numpy as np


class AveragePredictorCache(AbstractCache):

    # region Private variables

    __counters = None
    __trained_net = None
    __time_window = 0.0
    __processed_windows = 0
    __from_window_start = 0.0
    __priority_dict = None
    __update_sample_size = 0
    __prev_time = None

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self,
                 size: int,
                 counter_num: int,
                 time_window: float,
                 update_sample_size: int=5):
        """
        Construct a new AveragePredictorCache object.
        :param size: Size of cache.
        :param counter_num: Number of counters.
        :param time_window: Time window of one sketch activity.
        :param update_sample_size: How many items popularity to update when processing cache hit.
        """
        super().__init__(size)

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
        n = len(self.__counters)
        window_time = (time - self.__time_window * self.__processed_windows) / self.__time_window

        w = 1.0 / (n - 1 + window_time ** 2)
        w_0 = window_time ** 2 / (n - 1 + window_time ** 2)

        prediction_row = []
        for sketch in self.__counters[:-1]:
            frac = sketch.get_request_fraction(id_)
            prediction_row.append(frac * w)

        frac = self.__counters[-1].get_request_fraction(id_)
        prediction_row.append(frac * w_0)

        pop = float(np.sum(prediction_row))
        return pop

    def __update_time(self, time: float):
        """
        Updates time related activity - active sketches, time from window start, etc.
        :param time: Time of object arrival.
        """
        if self.__prev_time is None:
            self.__prev_time = time

        added_time = time - self.__prev_time
        assert added_time >= 0.0
        self.__prev_time = time

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

        pred_pop = self.__predict_pop(id_, time)
        self.__priority_dict[id_] = pred_pop

        sample = random.sample(self.__priority_dict.keys(), real_update_size)
        for i in sample:
            pred_pop = self.__predict_pop(i, time)
            self.__priority_dict[i] = pred_pop

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
