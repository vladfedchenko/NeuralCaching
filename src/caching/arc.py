"""
This module contains the implementation of ARC (Adaptive Replacement Cache).
"""
from caching.abstract_cache import AbstractCache, CachingObjectError
from helpers.collections import PriorityDict
from helpers.errors import AlgorithmError


class ARCache(AbstractCache):
    """
    ARCache implements cache with Adaptive Replacement policy.
    Inherits AbstractCache.
    """
    # region Private variables

    __T1 = None
    __T2 = None
    __B1 = None
    __B2 = None
    __p = 0
    __c = 0

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self, size):
        """
        Construct a new ARCache object.
        :param size: Size of cache.
        """
        super().__init__(size)
        self.__T1 = PriorityDict()
        self.__T2 = PriorityDict()
        self.__B1 = PriorityDict()
        self.__B2 = PriorityDict()
        self.__p = 0
        self.__c = size

    # endregion

    # region Private methods

    # REPLACE procedure from ARC paper
    def __replace(self, id_, time):
        if (not self.__T1.empty()) and ((len(self.__T1) > self.__p) or
                                        (id_ in self.__B2 and len(self.__T1) == self.__p)):
            rem = self.__T1.pop_smallest()
            self._remove_object(rem)
            self.__B1[rem] = time
        else:
            rem = self.__T2.pop_smallest()
            self._remove_object(rem)
            self.__B2[rem] = time

    # endregion

    # region Protected methods

    def _process_cache_hit(self, id_, size, time):
        """
        Move object to MRU in T2. Case I in ARC paper.
        :param id_: ID of the object.
        :param size: Size of the object.
        :param time: Time of the request.
        """
        if size != 1:
            raise CachingObjectError("Objects of size 1 only supported in ARC.")

        # ARC Case I
        if id_ in self.__T1:
            self.__T1[id_] = -1.0
            p = self.__T1.pop_smallest()
            assert p == id_
        self.__T2[id_] = time

    def _process_cache_miss(self, id_, size, time):
        """
        Implementing cases II-IV in ARC paper.
        :param id_: ID of the object
        :param size: Size of the object
        :param time: Time of the request
        """
        if size != 1:
            raise CachingObjectError("Objects of size 1 only supported in ARC.")

        # ARC Case II
        if id_ in self.__B1:
            if len(self.__B1) >= len(self.__B2):
                delta1 = 1
            else:
                delta1 = len(self.__B2) / len(self.__B1)
            self.__p = min(self.__p + delta1, self.__c)

            self.__replace(id_, time)

            self.__B1[id_] = -1.0
            p = self.__B1.pop_smallest()
            assert p == id_

            self.__T2[id_] = time
            self._store_object(id_, size)

        # ARC Case III
        elif id_ in self.__B2:
            if len(self.__B2) >= len(self.__B1):
                delta2 = 1
            else:
                delta2 = len(self.__B1) / len(self.__B2)
            self.__p = max(self.__p - delta2, 0)

            self.__replace(id_, time)

            self.__B2[id_] = -1.0
            p = self.__B2.pop_smallest()
            assert p == id_

            self.__T2[id_] = time
            self._store_object(id_, size)

        # ARC Case IV
        else:
            # Case A
            if len(self.__T1) + len(self.__B1) == self.__c:
                if len(self.__T1) < self.__c:
                    self.__B1.pop_smallest()
                    self.__replace(id_, time)
                else:
                    rem = self.__T1.pop_smallest()
                    self._remove_object(rem)

            # Case B
            elif len(self.__T1) + len(self.__B1) < self.__c:
                if len(self.__T1) + len(self.__B1) + len(self.__T2) + len(self.__B2) >= self.__c:
                    if len(self.__T1) + len(self.__B1) + len(self.__T2) + len(self.__B2) == 2 * self.__c:
                        self.__B2.pop_smallest()
                    self.__replace(id_, time)
            else:
                raise AlgorithmError("Should not happen according to ARC algorithm.")

            self._store_object(id_, size)
            self.__T1[id_] = time

    # endregion

    # region Public methods

    # endregion
