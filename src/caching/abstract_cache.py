"""
This module contains the definition of AbstractCache class and related errors.
"""


class AbstractCache(object):
    """
    AbstractCache encapsulates basic cache operations like save object, remove object, check is object is stored.
    It also manages cache storage (size, free storage).
    The policy to store or remove objects should be defined in subclasses.
    All subclasses need to override the request_object.
    """

    # region Private variables

    __cache_size = 0
    __free_cache = 0
    __saved_objects = None
    __last_req_time = 0

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self, size):
        """
        Construct a new AbstractCache object.
        :param size: size of cache
        """
        self.__cache_size = size
        self.__free_cache = size
        self.__saved_objects = {}
        self.__last_req_time = -1

    # endregion

    # region Private methods

    # endregion

    # region Protected methods

    def _store_object(self, id_, size):
        """
        Method to store an object in cache.
        :param id_: ID of the object
        :param size: Size of the object
        :raises NotEnoughStorage: If object size is larger than free storage of the cache
        :raises AlreadyStoredError: If object is already present in the cache
        """
        if id_ not in self.__saved_objects:
            if self.__free_cache >= size:
                self.__free_cache -= size
                self.__saved_objects[id_] = size
            else:
                raise NotEnoughStorage(f'Not enough storage for object with ID: {id_}, size: {size}')
        else:
            raise AlreadyStoredError(f'Cache already contains object with ID: {id_}')

    def _remove_object(self, id_):
        """
        Remove object from the cache.
        :param id_: ID of the object
        :raises ObjectNotSavedError: If the object is not stored in the cache
        """
        if id_ in self.__saved_objects:
            size = self.__saved_objects[id_]
            self.__free_cache += size
            del self.__saved_objects[id_]
        else:
            raise ObjectNotSavedError(f'Object with ID: {id_} not stored in the cache')

    def _is_cached(self, id_):
        """
        Check if object is cached.
        :param id_: ID of the object
        :return: bool -> True is object is cached, False otherwise
        """
        return id_ in self.__saved_objects

    @property
    def _free_cache(self):
        """
        Returns free cache size.
        :return: int -> Free cache size
        """
        return self.__free_cache

    def _process_cache_hit(self, id_, size, time):
        """
        Implement here all required to process cache hit.
        All subclasses need to implement this method.
        :param id_: ID of the object
        :param size: Size of the object
        :param time: Time of the request
        """
        raise NotImplementedError('Subclasses of AbstractCache need to implement __process_cache_hit')

    def _process_cache_miss(self, id_, size, time):
        """
        Implement here all required to process cache miss.
        All subclasses need to implement this method.
        :param id_: ID of the object
        :param size: Size of the object
        :param time: Time of the request
        """
        raise NotImplementedError('Subclasses of AbstractCache need to implement __process_cache_miss')

    # endregion

    # region Public methods

    def request_object(self, id_, size, time):
        """
        Method to request an object. Can be already stored in the cache or not.
        :param id_: ID of the object
        :param size: Size of the object
        :param time: Time of the request
        :return: bool -> True if cache hit, False if not
        :raises TimeOrderError: If the object is requested with less time than previous object (in the past)
        """
        if time > self.__last_req_time:
            self.__last_req_time = time
            if self._is_cached(id_):
                self._process_cache_hit(id_, size, time)
                return True
            else:
                self._process_cache_miss(id_, size, time)
                return False
        else:
            raise TimeOrderError('Wrong request time order: {0} <= {1}'.format(time, self.__last_req_time))

    # endregion


class ObjectNotSavedError(Exception):
    pass


class NotEnoughStorage(Exception):
    pass


class AlreadyStoredError(Exception):
    pass


class TimeOrderError(Exception):
    pass
