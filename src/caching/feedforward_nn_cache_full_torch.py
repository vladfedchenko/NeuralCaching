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

    __online = False
    __cf_coef = 0.0
    __past_dfs = []
    __past_pop = []
    __learning_rate = 0.0

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
                 update_sample_size: int=5,
                 online_learning: bool=False,
                 cf_coef: float=0.5,
                 learning_rate: float=0.001):
        """
        Construct a new FeedforwardNNCache object.
        :param size: Size of cache.
        :param trained_net: Trained neural network.
        :param counter_num: Number of counters.
        :param time_window: Time window of one sketch activity.
        :param update_sample_size: How many items popularity to update when processing cache hit.
        :param online_learning: Use online learning mechanism.
        :param cf_coef: Coefficient to prevent catastrophic forgetting while learning online.
        :param learning_rate: Learning rate for online learning.
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

        self.__online = online_learning
        self.__cf_coef = cf_coef
        assert 0.0 < self.__cf_coef < 1.0

        past_df_count = 1
        while self.__cf_coef ** past_df_count > 0.01:
            past_df_count += 1
        self.__past_dfs = [None] * past_df_count
        self.__past_pop = [None] * past_df_count
        self.__learning_rate = learning_rate

    # endregion

    # region Private methods

    def __predict_pop(self, id_, time: float, new_entry: bool) -> float:
        """
        Predict popularity of object using NN and sketches.
        :param id_: ID of the object.
        :param time: Time of arrival.
        :param new_entry: Prediction is for newly arrived object.
        :return: Predicted popularity.
        """
        prediction_row = []
        for sketch in self.__counters:
            frac = sketch.get_request_fraction(id_)
            frac = -np.log(frac + 10**-15)
            prediction_row.append(frac)

        window_time = (time - self.__time_window * self.__processed_windows) / self.__time_window
        prediction_row.append(window_time)

        np_matr = np.matrix([prediction_row])
        matr = torch.from_numpy(np_matr)
        pop_log = float(self.__trained_net(matr))
        pop = np.exp(-pop_log) - 10**-15

        if self.__online and new_entry:
            if self.__past_dfs[0] is None:
                self.__past_dfs[0] = []
                self.__past_pop[0] = []

            self.__past_dfs[0].append(prediction_row)
            self.__past_pop[0].append(id_)

        return pop

    def __learn_online(self):
        """
        Start online learning.
        """
        self.__past_dfs[0] = np.matrix(self.__past_dfs[0])
        self.__past_dfs[0] = torch.from_numpy(self.__past_dfs[0])

        self.__past_pop[0] = [self.__counters[-1].get_request_fraction(x) for x in self.__past_pop[0]]
        self.__past_pop[0] = torch.from_numpy(-np.log(np.matrix(self.__past_pop[0]).T + 10**-15))

        for i, inp in enumerate(self.__past_dfs):
            if inp is None:
                continue

            target = self.__past_pop[i]

            weight = self.__cf_coef**i

            self.__trained_net.backpropagation_learn(inp, target, self.__learning_rate, True, False, weight)

        for i in reversed(range(len(self.__past_dfs) - 1)):
            self.__past_pop[i + 1] = self.__past_pop[i]
            self.__past_dfs[i + 1] = self.__past_dfs[i]

        self.__past_dfs[0] = None
        self.__past_pop[0] = None

    def __update_time(self, time: float):
        """
        Updates time related activity - active sketches, time from window start, etc.
        :param time: Time of object arrival.
        """
        added_time = time - self.__time_window * self.__processed_windows
        self.__from_window_start += added_time

        while self.__from_window_start > self.__time_window:
            if self.__online:
                self.__learn_online()

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

        pred_pop = self.__predict_pop(id_, time, True)
        self.__priority_dict[id_] = pred_pop

        sample = random.sample(self.__priority_dict.keys(), real_update_size)
        for i in sample:
            pred_pop = self.__predict_pop(i, time, False)
            self.__priority_dict[i] = pred_pop

    def _process_cache_miss(self, id_, size, time):
        self.__update_time(time)
        self.__counters[-1].update_counters(id_)
        pred_pop = self.__predict_pop(id_, time, True)
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
