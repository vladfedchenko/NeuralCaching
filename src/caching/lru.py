"""
This module contains the implementation of LRU cache.
"""
from caching.abstract_cache import AbstractCache
from queue import PriorityQueue


class LRUCache(AbstractCache):
    """
    LRUCache implements cache with Least Recently Used policy.
    Inherits AbstractCache.
    """

    # region Private variables

    __last_access_map = None
    __access_queue = None
    __last_req_time = 0

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self, size):
        """
        Construct a new LRUCache object.
        :param size: size of cache
        """
        super().__init__(size)
        self.__last_access_map = {}
        self.__access_queue = PriorityQueue()
        self.__last_req_time = -1

    # endregion

    # region Private methods

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    def request_object(self, id_, size, time):
        """
        Method to request an object. Can be already stored in the cache or not.
        :param id_: ID of the object
        :param size: size of the object
        :param time: Time of the request
        :return: bool -> True if cache hit, False if not
        :raises TimeOrderError: if the object is requested with less time than previous object (in the past)
        :raises ObjectTooLargeError: if the object is too large to be stored in the cache
        """
        if time > self.__last_req_time:
            self.__last_req_time = time
            if self._is_cached(id_):
                self.__last_access_map[id_] = time
                return True
            else:
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

                return False

        else:
            raise TimeOrderError('Wrong request time order: {0} <= {1}'.format(time, self.__last_req_time))

    # endregion


class TimeOrderError(Exception):
    pass


class ObjectTooLargeError(Exception):
    pass
