"""
This module contains the implementation of LFU cache.
To count frequencies of the event occurrence count–min sketch is used.
"""
from caching.abstract_cache import AbstractCache
from queue import PriorityQueue
from helpers.collections import CountMinSketch


class LFUCache(AbstractCache):
    """
    LFUCache implements cache with Least Frequently Used policy.
    Inherits AbstractCache.
    """
    # region Private variables

    __min_sketch = None
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
        :param hash_func_num: Number of hash functions.
        """
        super().__init__(size)
        self.__min_sketch = CountMinSketch(count_min_cells, hash_func_num)
        self.__hit_queue = PriorityQueue()
        self.__hit_map = {}  # stores hit counts for cached objects only

    # endregion

    # region Private methods

    # endregion

    # region Protected methods

    def _process_cache_hit(self, id_, size, time, metadata):
        """
        If hit - update occurrence counters. Update hit map.
        :param id_: ID of the object.
        :param size: Size of the object.
        :param time: Time of the request.
        """
        self.__min_sketch.update_counters(id_)
        curr_freq = self.__min_sketch.get_count(id_)
        self.__hit_map[id_] = curr_freq

    def _process_cache_miss(self, id_, size, time, metadata):
        """
        Remove least frequently used object (LFU policy).
        Update counter for current object.
        Store requested object.
        :param id_: ID of the object.
        :param size: Size of the object.
        :param time: Time of the request.
        """
        self.__min_sketch.update_counters(id_)
        curr_freq = self.__min_sketch.get_count(id_)

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
