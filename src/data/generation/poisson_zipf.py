"""
This module contains the generator with poisson arrivals and Zipf popularity distribution.
"""


import numpy as np
from data.generation.abstract_generator import AbstractGenerator


class PoissonZipfGenerator(AbstractGenerator):
    """
    PoissonZipfGenerator implements generator with poisson arrivals and Zipf popularity distribution.
    Inherits AbstractGenerator.
    """
    # region Private variables

    __max_id = 0
    __poisson_lam = 0.0
    __zipf_param = 0.0
    __time_passed = 0

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self, max_id: int=100, poisson_lam: float=10, zipf_param: float=1.5):
        """
        Construct a new PoissonZipfGenerator object.
        :param max_id: Maximum ID of the object.
        :param poisson_lam: Poisson distribution parameter.
        :param zipf_param: Zipf distribution parameter.
        """
        self.__max_id = max_id
        self.__poisson_lam = poisson_lam
        self.__zipf_param = zipf_param
        self.__time_passed = 0

    # endregion

    # region Private methods

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    def next_item(self) -> (int, int, int):
        """
        Returns next generated item.
        :return: (int, int, int) -> Time from start, time from previous, item ID.
        """
        from_prev = np.random.poisson(self.__poisson_lam, 1)[0]
        id_ = np.random.zipf(self.__zipf_param, 1)[0]
        while id_ > self.__max_id:
            id_ = np.random.zipf(self.__zipf_param, 1)[0]
        self.__time_passed += from_prev
        return self.__time_passed, from_prev, id_

    def next_n_items(self, n: int) -> (int, int, int):
        """
        Returns next N generated item.
        :param n: Number of items to generate.
        :return: (int, int, int) -> Time from start, time from previous, item ID.
        """
        to_ret = []
        for _ in range(n):
            to_ret.append(self.next_item())
        return to_ret

    # endregion
