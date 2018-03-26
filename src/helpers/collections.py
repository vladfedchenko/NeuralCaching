"""
This module contains helping collections.
"""
import numpy as np
import math


class CollectionError(Exception):
    pass


class CollectionEmptyError(CollectionError):
    pass


class DoubleLinkedList:
    # region Private variables

    class __Node:
        """
        This class represent a node of a DoubleLinkedList.
        """
        prev = None
        next_ = None
        val = None
        father = None

        def __init__(self, prev, next_, val, father):
            self.prev = prev
            self.next_ = next_
            self.val = val
            self.father = father

    class WrongListError(CollectionError):
        pass

    __head = None
    __tail = None
    __len = 0

    # endregion__len

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self):
        """
        Construct a new DoubleLinkedList object.
        """
        self.__head = None
        self.__tail = None
        self.__len = 0
        pass

    # endregion

    # region Private methods

    def __len__(self):
        """
        Returns the length of the list.
        :return: int -> Length of the list.
        """
        return self.__len

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    @property
    def head(self):
        """
        :return: DoubleLinkedList.__Node -> Head node of the list.
        """
        return self.__head

    @property
    def tail(self):
        """
        :return: DoubleLinkedList.__Node -> Tail node of the list.
        """
        return self.__tail

    @property
    def length(self):
        """
        Returns the length of the list.
        :return: int -> Length of the list.
        """
        return self.__len

    def empty(self):
        """
        Check if the list is empty.
        :return: bool -> True if empty, False otherwise.
        """
        return len(self) == 0

    def put_head(self, val):
        """
        To add a new node and make it head of the list.
        :param val: Value of the node.
        :return: DoubleLinkedList.__Node -> Newly createCollectionEmptyErrord node.
        """
        node = self.__Node(None, self.__head, val, self)
        if self.__head is not None:
            self.__head.prev = node
        self.__head = node
        self.__len += 1
        if self.__len == 1:
            self.__tail = self.__head
        return self.__head

    def put_tail(self, val):
        """
        To add a new node and make it tail of the list.
        :param val: Value of the node.
        :return: DoubleLinkedList.__Node -> Newly created node.
        """
        node = self.__Node(self.__tail, None, val, self)
        if self.__tail is not None:
            self.__tail.next_ = node
        self.__tail = node
        self.__len += 1
        if self.__len == 1:
            self.__head = self.__tail
        return self.__tail

    def pop_head(self):
        """
        Get the value of the head and remove head from the list.__len
        :return: Value of the head node. Error if the list is empty.
        :raises CollectionEmptyError: If the collection is empty.
        """
        if self.__head is None:
            raise CollectionEmptyError("The collection is empty!")
        else:
            ret = self.__head.val
            self.__head = self.__head.next_
            if self.__head is not None:
                self.__head.prev = None
            self.__len -= 1
            return ret

    def pop_tail(self):
        """
        Get the value of the head and remove head from the list.
        :return: Value of the head node. None if the list is empty.
        :raises CollectionEmptyError: If the collection is empty.
        """
        if self.__tail is None:
            raise CollectionEmptyError("The collection is empty!")
        else:
            ret = self.__tail.val
            self.__tail = self.__head.prev
            if self.__tail is not None:
                self.__tail.next = None
            self.__len -= 1
            return ret

    def remove_from_list(self, node):
        """
        Removes passed node from the list.
        :param node: Node to be removed.
        :raises WrongListError: When trying to remove node that belongs to the other list.
        """
        if node.father is not self:
            raise self.WrongListError("Passed node does not belong to the list.")
        if node is self.__head:
            self.pop_head()
        elif node is self.__tail:
            self.pop_tail()
        else:
            node.prev.next_ = node.next_
            node.next_.prev = node.prev
            self.__len -= 1

    # endregion


class LRUQueue:
    """
    Queue that implements LRU policy.
    No repeating values allowed.
    """
    # region Private variables

    class RepeatingValueError(CollectionError):
        pass

    __linked_list = None
    __node_map = None

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self):
        """
        Construct a new LRUQueue object.
        """
        self.__linked_list = DoubleLinkedList()
        self.__node_map = {}

    def __contains__(self, item):
        """
        Check if item is in the queue.
        """
        return item in self.__node_map

    # endregion

    # region Private methods

    def __len__(self):
        """
        Returns the length of the queue.
        :return: int -> Length of the queue.
        """
        return len(self.__linked_list)

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    @property
    def length(self):
        """
        Returns the length of the queue.
        :return: int -> Length of the queue.
        """
        return len(self)

    def empty(self):
        """
        Check if the queue is empty.
        :return: bool -> True if empty, False otherwise.
        """
        return self.__linked_list.empty()

    def put(self, item):
        """
        Put the item in the queue.
        :param item: Item to put in the queue.
        :raises RepeatingValueError: If repeating values detected.
        """
        if item in self.__node_map:
            raise self.RepeatingValueError("Item {0} is already stored in the queue.".format(item))
        node = self.__linked_list.put_tail(item)
        self.__node_map[item] = node

    def pop(self):
        """
        Get least recently used item.
        :return: Least recently used item.
        """
        item = self.__linked_list.pop_head()
        del self.__node_map[item]
        return item

    def update(self, item):
        """
        Update the item to make it most recent user (MRU).
        :param item: Item to update if it is in the queue. Insert otherwise.
        """
        if item not in self:
            self.put(item)
        else:
            node = self.__node_map[item]
            self.__linked_list.remove_from_list(node)
            self.__node_map[item] = self.__linked_list.put_tail(item)

    def remove(self, item):
        """
        Forcefully remove item from the queue.
        :param item: Item to remove.
        """
        self.__linked_list.remove_from_list(self.__node_map[item])
        del self.__node_map[item]

    # endregion


class CountMinSketch:
    """
    A count–min sketch implementation.
    """
    # region Private variables

    __count_min_cells = 0
    __min_sketch_buckets = None
    __hash_additions = None

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self, count_min_cells=100, hash_func_num=10):
        """
        Construct a new CountMinSketch object.
        :param count_min_cells: Number of cells for count min.
        :param hash_func_num: Number of hash functions.
        """
        self.__count_min_cells = count_min_cells
        self.__min_sketch_buckets = np.matrix([[0] * count_min_cells for _ in range(hash_func_num)])
        self.__hash_additions = [str(hash(str(i))) for i in range(hash_func_num)]

    @staticmethod
    def construct_by_constraints(overcount_perc: float):
        """
        Construct a count-min sketch instance by epsilon constraint.
        :param overcount_perc: Maximum allowed overcount of item over the number of requests.
        :return: An instance of CountMinSketch
        """
        epsilon = overcount_perc
        cells = 1.0 / epsilon
        hashes = math.log(epsilon, 0.5)
        return CountMinSketch(int(cells), int(hashes))

    # endregion

    # region Private methods

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    def update_counters(self, id_):
        """
        Update counters of count-min sketch.
        :param id_: ID of the object.
        """
        for i, hash_add in enumerate(self.__hash_additions):
            hash_key = hash_add + str(id_)
            hash_val = hash(hash_key)
            cell = hash_val % self.__count_min_cells
            self.__min_sketch_buckets[i, cell] += 1

    def get_count(self, id_) -> int:
        """
        Get access count using count-min sketch
        :param id_: ID of the object.
        :return: Estimated hit count.
        """
        counts = []
        for i, hash_add in enumerate(self.__hash_additions):
            hash_key = hash_add + str(id_)
            hash_val = hash(hash_key)
            cell = hash_val % self.__count_min_cells
            counts.append(self.__min_sketch_buckets[i, cell])

        return np.min(counts)

    def get_counter_state(self) -> np.ndarray:
        """
        :return: The state of count-min sketch as a flat array.
        """
        return self.__min_sketch_buckets.flatten()

    # endregion
