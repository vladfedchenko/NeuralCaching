"""
This module contains the generator with Poisson arrivals and Zipf popularity distribution and variations of this
generator - disappearing population generator and shuffled popularity generator.
"""
import numpy as np
import random
from data.generation import AbstractTimedGenerator, AbstractDistributionGenerator, ZipfGenerator


class PoissonZipfGenerator(AbstractDistributionGenerator, AbstractTimedGenerator):
    """
    PoissonZipfGenerator implements generator with poisson arrivals and Zipf popularity distribution.
    Inherits AbstractTimedGenerator.
    """

    # region Private variables

    __zipf_generator = None
    __poisson_lam = 0.0
    __time_passed = 0

    # endregion

    # region Protected variables

    _id_shift = 0

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self, max_id: int=1000, poisson_lam: float=10, zipf_param: float=1.0, id_shift: int=0):
        """
        Construct a new PoissonZipfGenerator object.
        :param max_id: Maximum ID of the object.
        :param poisson_lam: Poisson distribution parameter.
        :param zipf_param: Zipf distribution parameter.
        :param id_shift: Shift of the starting item ID.
        """
        self.__zipf_generator = ZipfGenerator(zipf_param, max_id)
        self.__poisson_lam = poisson_lam
        self.__time_passed = 0

        self._id_shift = id_shift

    # endregion

    # region Private methods

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    def get_distribution_map(self) -> "dict, None":
        """
        Return distribution map. Maps item IDs to the probability to observe it.
        :return: Distribution map or None.
        """
        return self.__zipf_generator.get_distribution_map()

    def next_item(self) -> (int, int, int):
        """
        Returns next generated item.
        :return: (int, int, int) -> Time from start, time from previous, item ID.
        """
        from_prev = np.random.poisson(self.__poisson_lam, 1)[0]
        id_ = self.__zipf_generator.get_item() + self._id_shift
        self.__time_passed += from_prev
        return self.__time_passed, from_prev, id_

    # endregion


class DisappearingPoissonZipfGenerator(PoissonZipfGenerator):
    """
    The generator with poisson arrivals and Zipf popularity distribution in which objects disappear accordingly to
    some Poisson distribution and reappear accordingly to other Poisson distribution.
    """
    # region Private variables

    __poisson_disappear = 0
    __poisson_reappear = 0
    __disappear_map = None

    __last_time = 0

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self,
                 max_id: int=1000,
                 poisson_lam: float=10,
                 zipf_param: float=1.0,
                 id_shift: int = 0,
                 poisson_disappear: float=10**6,
                 poisson_reappear: float=10**6,
                 start_randomized: bool=False):
        """
        Construct a new DisappearingPoissonZipfGenerator object.
        :param max_id: Maximum ID of the object.
        :param poisson_lam: Poisson distribution parameter.
        :param zipf_param: Zipf distribution parameter.
        :param id_shift: Shift of the starting item ID.
        :param poisson_disappear: Poisson distribution parameter of disappearing.
        :param poisson_reappear: Poisson distribution parameter of reappearing.
        """
        super().__init__(max_id, poisson_lam, zipf_param, id_shift)

        self.__poisson_disappear = poisson_disappear
        self.__poisson_reappear = poisson_reappear

        if start_randomized:
            self.__disappear_map = {i: [np.random.poisson(self.__poisson_disappear, 1)[0], random.random() < 0.5]
                                    for i in range(id_shift + 1, id_shift + max_id + 1)}
        else:
            self.__disappear_map = {i: [np.random.poisson(self.__poisson_disappear, 1)[0], False]
                                    for i in range(id_shift + 1, id_shift + max_id + 1)}

        self.__last_time = 0

    # endregion

    # region Private methods

    def __check_now_valid(self, from_start: int, id_: int) -> bool:
        """
        To check if item became available. Also updates __disappear_map.
        :param from_start: Time of item arrival.
        :param id_: ID of the object.
        :return: True if object now available, False otherwise.
        """
        cur_status = self.__disappear_map[id_][1]
        while self.__disappear_map[id_][0] <= from_start:
            if cur_status:  # was missing, appeared
                self.__disappear_map[id_][0] += np.random.poisson(self.__poisson_disappear, 1)[0]
            else:  # was available, disappeared
                self.__disappear_map[id_][0] += np.random.poisson(self.__poisson_reappear, 1)[0]
            cur_status = not cur_status
        self.__disappear_map[id_][1] = cur_status
        return not cur_status

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    def get_distribution_map(self) -> "dict, None":
        """
        Return distribution map. Maps item IDs to the probability to observe it.
        :return: Distribution map or None.
        """
        return None

    def next_item(self) -> (int, int, int):
        """
        Returns next generated item.
        :return: (int, int, int) -> Time from start, time from previous, item ID.
        """
        found_valid = False
        from_start, from_prev, id_ = (None, None, None)

        while not found_valid:
            from_start, from_prev, id_ = super().next_item()
            if from_start >= self.__disappear_map[id_][0]:  # item status changed: reappeared or disappeared
                found_valid = self.__check_now_valid(from_start, id_)
            elif not self.__disappear_map[id_][1]:  # item status remains the same and it is not disappeared
                found_valid = True

        from_prev = from_start - self.__last_time
        self.__last_time = from_start

        return from_start, from_prev, id_

    # endregion


class PoissonShuffleZipfGenerator(PoissonZipfGenerator):
    """
    PoissonShuffleZipfGenerator implements generator with poisson arrivals and Zipf popularity distribution,
    but every time window item popularity is randomly shuffled.
    Inherits AbstractTimedGenerator.
    """
    # region Private variables

    __shuffle_window = 0.0
    __next_shuffle = 0.0

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self,
                 max_id: int=1000,
                 poisson_lam: float=10,
                 zipf_param: float=1.0,
                 id_shift: int=0,
                 shuffle_window: float=1000.0,
                 start_shuffled: bool=False):
        """
        Construct a new PoissonShuffleZipfGenerator object.
        :param max_id: Maximum ID of the object.
        :param poisson_lam: Poisson distribution parameter.
        :param zipf_param: Zipf distribution parameter.
        :param id_shift: Shift of the starting item ID.
        :param shuffle_window: Time window after which popularity if shuffled.
        :param start_shuffled: Determine if the items are already shuffled during first time window.
        """
        super().__init__(max_id, poisson_lam, zipf_param, id_shift)

        self.__shuffle_window = shuffle_window
        self.__next_shuffle = shuffle_window

        self.__shuffle_map = np.arange(id_shift + 1, id_shift + max_id + 1)
        if start_shuffled:
            np.random.shuffle(self.__shuffle_map)

    # endregion

    # region Private methods

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    def get_distribution_map(self) -> "dict, None":
        """
        Return distribution map. Maps item IDs to the probability to observe it.
        :return: Distribution map or None.
        """
        vals = list(super().get_distribution_map().values())
        probability = np.mean(vals)
        return {i: probability for i in self.__shuffle_map}

    def next_item(self) -> (int, int, int):
        """
        Returns next generated item.
        :return: (int, int, int) -> Time from start, time from previous, item ID.
        """
        from_start, from_prev, id_ = super().next_item()
        while from_start > self.__next_shuffle:
            self.__next_shuffle += self.__shuffle_window
            np.random.shuffle(self.__shuffle_map)

        index = id_ - self._id_shift - 1
        return from_start, from_prev, self.__shuffle_map[index]

    # endregion
