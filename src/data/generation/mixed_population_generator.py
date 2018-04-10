"""
This module contains a generates a trace with mixed population.
Both have Poisson arrivals and Zipf popularity but in one items randomly disappear and reappear accordingly to
some other Poisson distribution.
"""
from data.generation import AbstractTimedGenerator
import random


class MixedPopulationTimedGenerator(AbstractTimedGenerator):
    """
    A generator that generates a trace with mixed population.
    Both have Poisson arrivals and Zipf popularity but in one items randomly disappear and reappear accordingly to
    some other Poisson distribution.
    """
    # region Private variables

    __fst_generator = None
    __second_generator = None
    __last_time = 0
    __last_item_first = None
    __last_item_second = None
    __same_time_spread = 10 ** (-10)

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self,
                 first_generator: AbstractTimedGenerator,
                 second_generator: AbstractTimedGenerator,
                 same_time_spread: float=10**(-10)):
        """
        Create a new instance of MixedPopulationTimedGenerator
        :param first_generator: First item generator.
        :param second_generator: Second item generator.
        """
        self.__fst_generator = first_generator
        self.__second_generator = second_generator
        self.__last_time = 0.0
        self.__last_item_first = self.__fst_generator.next_item()
        self.__last_item_second = self.__second_generator.next_item()
        self.__same_time_spread = same_time_spread

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
        if self.__last_item_first[0] == self.__last_item_second[0]:  # two items arrive at the same time
            prob = random.random()
            if prob > 0.5:
                to_ret = self.__last_item_second
                self.__last_item_second = self.__second_generator.next_item()
            else:
                to_ret = self.__last_item_first
                self.__last_item_first = self.__fst_generator.next_item()

        elif self.__last_item_first[0] < self.__last_item_second[0]:  # regular population item arrived earlier
            to_ret = self.__last_item_first
            self.__last_item_first = self.__fst_generator.next_item()

        else:
            to_ret = self.__last_item_second
            self.__last_item_second = self.__second_generator.next_item()

        if to_ret[0] > self.__last_time:
            from_prev = float(to_ret[0] - self.__last_time)
            self.__last_time = float(to_ret[0])
            return self.__last_time, from_prev, to_ret[2]

        else:
            self.__last_time += self.__same_time_spread
            return self.__last_time, self.__same_time_spread, to_ret[2]

    # endregion
