"""
This module contains a generates a trace with mixed population.
Both have Poisson arrivals and Zipf popularity but in one items randomly disappear and reappear accordingly to
some other Poisson distribution.
"""
from data.generation import AbstractGenerator, PoissonZipfGenerator, DisappearingPoissonZipfGenerator
import random


class MixedPoissonZipfGenerator(AbstractGenerator):
    """
    A generator that generates a trace with mixed population.
    Both have Poisson arrivals and Zipf popularity but in one items randomly disappear and reappear accordingly to
    some other Poisson distribution.
    """
    # region Private variables

    __reg_generator = None
    __dis_generator = None
    __last_time = 0
    __last_item_reg = None
    __last_item_dis = None

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self,
                 poisson_reg: float=10.0,
                 zipf_reg: float=1.0,
                 max_id_reg=1000,
                 poisson_dis: float=10.0,
                 zipf_dis: float=1.0,
                 max_id_dis=1000,
                 poisson_disappear: float = 10 ** 6,
                 poisson_reappear: float = 10 ** 6,
                 id_shift: int=0):
        """

        :param poisson_reg: Poisson distribution parameter for regular population.
        :param zipf_reg: Zipf distribution parameter for regular population.
        :param max_id_reg: Maximum ID of the object for regular population.
        :param poisson_dis: Poisson distribution parameter for disappearing population.
        :param zipf_dis: Zipf distribution parameter for disappearing population.
        :param max_id_dis: Maximum ID of the object for disappearing population.
        :param poisson_disappear: Poisson distribution parameter of disappearing.
        :param poisson_reappear: Poisson distribution parameter of reappearing.
        :param id_shift: Shift of the starting item ID.
        """
        self.__reg_generator = PoissonZipfGenerator(max_id_reg, poisson_reg, zipf_reg, id_shift)
        self.__dis_generator = DisappearingPoissonZipfGenerator(max_id_dis,
                                                                poisson_dis,
                                                                zipf_dis,
                                                                id_shift + max_id_reg,
                                                                poisson_disappear,
                                                                poisson_reappear)
        self.__last_time = 0
        self.__last_item_reg = self.__reg_generator.next_item()
        self.__last_item_dis = self.__dis_generator.next_item()

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
        if self.__last_item_reg[0] == self.__last_item_dis[0]:  # two items arrive at the same time
            prob = random.random()
            if prob > 0.5:
                to_ret = self.__last_item_dis
                self.__last_item_dis = self.__dis_generator.next_item()
            else:
                to_ret = self.__last_item_reg
                self.__last_item_reg = self.__reg_generator.next_item()

        elif self.__last_item_reg[0] < self.__last_item_dis[0]:  # regular population item arrived earlier
            to_ret = self.__last_item_reg
            self.__last_item_reg = self.__reg_generator.next_item()

        else:
            to_ret = self.__last_item_dis
            self.__last_item_dis = self.__dis_generator.next_item()

        if to_ret[0] > self.__last_time:
            from_prev = to_ret[0] - self.__last_time
            self.__last_time = to_ret[0]
            return to_ret[0], from_prev, to_ret[2]

        else:
            self.__last_time += 1
            return self.__last_time, 1, to_ret[2]

    # endregion
