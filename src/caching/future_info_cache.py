"""
This caching policy mimics neural-network based policy. But instead of predicting the popularity by a NN it takes a
trace with precalculated popularity in the future.
"""
from caching.abstract_cache import AbstractCache
from helpers.collections import PriorityDict
import pandas as pd
import random


class FutureInfoCache(AbstractCache):

    # region Private variables

    __counters = None
    __trace_iter = None
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
                 path_to_mod_trace: str):
        """
        Construct a new FeedforwardNNCache object.
        :param size: Size of cache.
        :param path_to_mod_trace: Path to modified trace CSV file.
        :param update_sample_size: How many items popularity to update when processing cache hit.
        """
        super().__init__(size)

        input_df = pd.read_csv(path_to_mod_trace, header=None, names=["from_start", "from_prev", "id", "future_pop"])
        self.__trace_iter = input_df.iterrows()

        self.__priority_dict = PriorityDict()

    # endregion

    # region Private methods

    def __predict_pop(self, id_, time: float) -> float:
        """
        Fetch the popularity from the modified trace.
        :param id_: ID of the object.
        :param time: Time of arrival.
        :return: Fetched popularity.
        """
        _, row = next(self.__trace_iter)
        assert int(row.id) == int(id_)
        return row.future_pop

    # endregion

    # region Protected methods

    def _process_cache_hit(self, id_, size, time, metadata):
        if len(self.__priority_dict) < self.__update_sample_size:
            real_update_size = len(self.__priority_dict)
        else:
            real_update_size = self.__update_sample_size

        pred_pop = self.__predict_pop(id_, time)
        self.__priority_dict[id_] = pred_pop

        # sample = random.sample(self.__priority_dict.keys(), real_update_size)
        # for id_ in sample:
        #     pred_pop = self.__predict_pop(id_, time)
        #     self.__priority_dict[id_] = pred_pop

    def _process_cache_miss(self, id_, size, time, metadata):
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
