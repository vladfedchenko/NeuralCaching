"""
This module contains some helpers to ease the creation of others modules.
"""
from queue import PriorityQueue


class PriorityQueueUniqueUpdatable:
    """
    PriorityQueueUniqueUpdatable implements a priority queue with an ability to update priority keys.
    For unique values only.
    """
    # region Private variables

    __priority_queue = None
    __access_map = None

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self):
        """
        Construct a new PriorityQueueUniqueUpdatable object.
        """
        self.__priority_queue = PriorityQueue()
        self.__access_map = {}

    # endregion

    # region Private methods

    def __len__(self):
        """
        Length of the queue.
        :return: int -> Length of the queue.
        """
        return self.__priority_queue.qsize()

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    def put(self, kv):
        """
        Puts key-value tuple in the priority queue.
        :param kv: Key-value tuple. Key determines the priority.
        :raises RepeatingValueError: If repeating values are encountered.
        """
        if kv[1] in self.__access_map:
            raise RepeatingValueError("PriorityQueueUniqueUpdatable allows only unique values")
        self.__priority_queue.put(kv)
        self.__access_map[kv[1]] = kv[0]

    def get(self):
        """
        Get item with least priority key.
        :return: Key and value according to priority.
        """
        k, v = self.__priority_queue.get()
        while k != self.__access_map[v]:
            self.__priority_queue.put((self.__access_map[v], v))
            k, v = self.__priority_queue.get()
        del self.__access_map[v]
        return k, v

    def update(self, kv):
        """
        To update the priority of some item. If the item is not present - it will be inserted into the queue.
        :param kv: Key-value tuple. Key determines the priority.
        """
        if not kv[1] in self.__access_map:
            self.put(kv)
        else:
            self.__access_map[kv[1]] = kv[0]

    def empty(self):
        """
        Check if the queue is empty.
        :return: bool -> True if empty, False otherwise.
        """
        return self.__priority_queue.empty()

    # endregion


class RepeatingValueError(Exception):
    pass
