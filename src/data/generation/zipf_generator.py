"""
This module contains the implementation of a generator that generates items accordingly to Zipf's law.
"""
from data.generation.abstract_generator import AbstractDistributionGenerator, AbstractGenerator
import math
import random
import bisect


class ZipfGenerator(AbstractGenerator, AbstractDistributionGenerator):
    """
    ZipfGenerator is the implementation of a generator that generates items accordingly to Zipf's law.
    """
    # region Private variables

    __cdf_mapping = None

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self, alpha: float, n: int):
        """
        Construct a new ZipfGenerator object.
        :param alpha: Parameter of the distribution >= 0.
        :param n: Maximum item to be drawn.
        """
        # Calculate generalized harmonic numbers of order n of lambda:
        tmp = [1.0 / (math.pow(float(i), alpha)) for i in range(1, n + 1)]
        zeta = [0.0]
        for t in tmp:
            zeta.append(zeta[-1] + t)

        self.__cdf_mapping = [x / zeta[-1] for x in zeta]

    # endregion

    # region Private methods

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    def get_distribution_map(self) -> "dict, None":
        """
        Return distribution map. Maps item IDs to the probability to observe it.
        :return: Distribution map.
        """
        n = len(self.__cdf_mapping) - 1
        dist_map = {}
        for id_ in range(1, n + 1):
            dist_map[id_] = self.__cdf_mapping[id_] - self.__cdf_mapping[id_ - 1]
        return dist_map

    def get_item(self):
        """
        Returns next generated item.
        :return: int -> Next generated item.
        """
        u = random.random()  # uniform [0.0, 1.0)
        return bisect.bisect(self.__cdf_mapping, u)

    # endregion
