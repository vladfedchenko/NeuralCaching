"""
This module contains the implementation of LFU cache.
To count frequencies of the event occurrence count–min sketch is used.
"""
from caching.abstract_cache import AbstractCache
import numpy as np
from queue import PriorityQueue


class LFUCache(AbstractCache):

    # region Private variables

    __count_min_cells = 0
    __hash_func_num = 0
    __min_sketch_counter = None
    __hash_additions = None
    __hit_queue = None
    __hit_map = {}

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self, size, count_min_cells=100, hash_func_num=10):
        """
        Construct a new LFUCache object.
        :param size: Size of cache.
        :param count_min_cells: Number of cells for count min.
        """
        super().__init__(size)
        self.__count_min_cells = count_min_cells
        self.__hash_func_num = hash_func_num
        self.__min_sketch_counter = np.array([0] * count_min_cells)

        self.__hash_additions = [str(hash(str(i))) for i in range(hash_func_num)]

        self.__hit_queue = PriorityQueue()
        self.__hit_map = {}  # stores hit counts for cached objects only

    # endregion

    # region Private methods

    def __update_counters(self, id_):
        """
        Update counters of count-min sketch.
        :param id_: ID of the object.
        """
        for hash_add in self.__hash_additions:
            hash_key = hash_add + str(id_)
            hash_val = hash(hash_key)
            cell = hash_val % self.__count_min_cells
            self.__min_sketch_counter[cell] += 1

    def __get_count(self, id_):
        """
        Get access count using count-min sketch
        :param id_: ID of the object.
        :return: Estimated hit count.
        """
        indices = []
        for hash_add in self.__hash_additions:
            hash_key = hash_add + str(id_)
            hash_val = hash(hash_key)
            cell = hash_val % self.__count_min_cells
            indices.append(cell)

        counts = self.__min_sketch_counter[indices]
        return np.min(counts)

    # endregion

    # region Protected methods

    def _process_cache_hit(self, id_, size, time):
        """
        If hit - update occurrence counters. Update hit map.
        :param id_: ID of the object.
        :param size: Size of the object.
        :param time: Time of the request.
        """
        self.__update_counters(id_)
        curr_freq = self.__get_count(id_)
        self.__hit_map[id_] = curr_freq

    def _process_cache_miss(self, id_, size, time):
        """
        Remove least frequently used object (LFU policy).
        Update counter for current object.
        Store requested object.
        :param id_: ID of the object.
        :param size: Size of the object.
        :param time: Time of the request.
        """
        self.__update_counters(id_)
        curr_freq = self.__get_count(id_)

        if self._free_cache < size and not self.__hit_queue.empty():
            c, i = self.__hit_queue.get()  # counters are not updated in queue, need to compensate for it
            while self.__hit_map[i] != c:
                self.__hit_queue.put((self.__hit_map[i], i))
                c, i = self.__hit_queue.get()

            self._remove_object(i)  # removing old from cache
            del self.__hit_map[i]

        self._store_object(id_, size)  # storing new to cache
        self.__hit_map[id_] = curr_freq
        self.__hit_queue.put((curr_freq, id_))

    # endregion

    # region Public methods

    # endregion
