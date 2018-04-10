"""
This module contains the implementation of abstract generator.
"""


class AbstractGenerator:
    # region Private variables

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    # endregion

    # region Private methods

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    def get_item(self) -> int:
        """
        Generate an item.
        :return: ID of the item
        """
        raise NotImplementedError()

    def next_n_items(self, n) -> "list":
        """
        Generate N items.
        :param n: Number of items to be generated.
        """
        to_ret = []
        for _ in range(n):
            to_ret.append(self.get_item())
        return to_ret

    # endregion


class AbstractTimedGenerator:
    """
    Abstract generator implementation.
    """
    # region Private variables

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    # endregion

    # region Private methods

    def __next__(self):
        """
        Returns next generated item.
        """
        return self.next_item()

    # endregion

    # region Protected methods

    # endregion

    # region Public methods

    def next_item(self) -> "(float, float, int), (int, int, int)":
        """
        Returns next generated item.
        All subclasses need to implement this method.
        """
        raise NotImplementedError()

    def next_n_items(self, n) -> "list":
        """
        Returns next N generated item.
        :param n: Number of items to be generated.
        """
        to_ret = []
        for _ in range(n):
            to_ret.append(self.next_item())
        return to_ret

    # endregion


class AbstractDistributionGenerator:

    # region Private variables

    # endregion

    # region Protected variables

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

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
        raise NotImplementedError()

    # endregion
