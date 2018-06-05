"""
This module contains the implementation of LRU cache.
"""
from caching.abstract_cache import AbstractCache, NotEnoughStorage
from helpers.collections import PriorityDict


class LRUCache(AbstractCache):
    """
    LRUCache implements cache with Least Recently Used policy.
    Inherits AbstractCache.
    """

    # region Private variables

    __access_priority_dict = None

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self, size):
        """
        Construct a new LRUCache object.
        :param size: Size of cache.
        """
        super().__init__(size)
        self.__access_priority_dict = PriorityDict()

    # endregion

    # region Private methods

    # endregion

    # region Protected methods

    def _process_cache_hit(self, id_, size, time):
        """
        Only update last access time for cached objects.
        :param id_: ID of the object.
        :param size: Size of the object.
        :param time: Time of the request.
        """
        self.__access_priority_dict[id_] = time

    def _process_cache_miss(self, id_, size, time):
        """
        Remove oldest accessed object (LRU policy). Store requested object.
        :param id_: ID of the object.
        :param size: Size of the object.
        :param time: Time of the request.
        :raises ObjectTooLargeError: If the object is too large to be stored in the cache.
        """
        free = self._free_cache

        while free < size:
            if len(self.__access_priority_dict) == 0:
                raise NotEnoughStorage(f'Cache cannot hold object of size {size}')

            i = self.__access_priority_dict.pop_smallest()
            self._remove_object(i)
            free = self._free_cache

        self.__access_priority_dict[id_] = time
        self._store_object(id_, size)

    # endregion

    # endregion
