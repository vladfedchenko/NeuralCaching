"""
This module contains the generator with poisson arrivals and Zipf popularity distribution.
"""
import numpy as np
from data.generation import AbstractGenerator, ZipfGenerator


class PoissonZipfGenerator(AbstractGenerator):
    """
    PoissonZipfGenerator implements generator with poisson arrivals and Zipf popularity distribution.
    Inherits AbstractGenerator.
    """
    # region Private variables

    __zipf_generator = None
    __poisson_lam = 0.0
    __time_passed = 0

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self, max_id: int=1000, poisson_lam: float=10, zipf_param: float=1.0):
        """
        Construct a new PoissonZipfGenerator object.
        :param max_id: Maximum ID of the object.
        :param poisson_lam: Poisson distribution parameter.
        :param zipf_param: Zipf distribution parameter.
        """
        self.__zipf_generator = ZipfGenerator(zipf_param, max_id)
        self.__poisson_lam = poisson_lam
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
        id_ = self.__zipf_generator.next_item()
        self.__time_passed += from_prev
        return self.__time_passed, from_prev, id_

    # endregion


class DisappearingPoissonZipfGenerator(PoissonZipfGenerator):
    """
    The generator with poisson arrivals and Zipf popularity distribution in which objects disappear accordingly to
    some Poisson distribution and reappear accordingly to other Poisson distribution.
    """
    # region Private variables

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
                 poisson_disappear=10000,
                 poisson_reappear=10000):
        """
        Construct a new DisappearingPoissonZipfGenerator object.
        :param max_id: Maximum ID of the object.
        :param poisson_lam: Poisson distribution parameter.
        :param zipf_param: Zipf distribution parameter.
        :param poisson_disappear: Poisson distribution parameter of disappearing.
        :param poisson_reappear: Poisson distribution parameter of reappearing.
        """
        super().__init__(max_id, poisson_lam, zipf_param)

        # TODO: finish implementation.

    # endregion

    # region Private methods

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    # endregion
