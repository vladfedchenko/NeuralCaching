"""
This module contains the implementation of LRU cache.
"""
from caching.abstract_cache import AbstractCache, ObjectTooLargeError
from queue import PriorityQueue


class LRUCache(AbstractCache):
    """
    LRUCache implements cache with Least Recently Used policy.
    Inherits AbstractCache.
    """

    # region Private variables

    __last_access_map = None
    __access_queue = None

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self, size):
        """
        Construct a new LRUCache object.
        :param size: Size of cache
        """
        super().__init__(size)
        self.__last_access_map = {}
        self.__access_queue = PriorityQueue()

    # endregion

    # region Private methods

    # endregion

    # region Protected methods

    def _process_cache_hit(self, id_, size, time):
        """
        Only update last access time for cached objects.
        :param id_: ID of the object
        :param size: Size of the object
        :param time: Time of the request
        """
        self.__last_access_map[id_] = time

    def _process_cache_miss(self, id_, size, time):
        """
        Remove oldest accessed object (LRU policy). Store requested object.
        :param id_: ID of the object
        :param size: Size of the object
        :param time: Time of the request
        :raises ObjectTooLargeError: If the object is too large to be stored in the cache
        """
        free = self._free_cache

        while free < size:
            if self.__access_queue.empty():
                raise ObjectTooLargeError(f'Cache cannot hold object of size {size}')

            t, i = self.__access_queue.get()
            if self.__last_access_map[i] > t:  # object has been requested after it was added to queue
                self.__access_queue.put((self.__last_access_map[i], i))
            else:
                self._remove_object(i)
                del self.__last_access_map[i]

            free = self._free_cache

        self.__last_access_map[id_] = time
        self.__access_queue.put((time, id_))
        self._store_object(id_, size)

    # endregion

    # endregion
