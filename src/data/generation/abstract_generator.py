"""
This module contains the implementation of abstract generator.
"""


class AbstractGenerator:
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

    def next_item(self):
        """
        Returns next generated item.
        All subclasses need to implement this method.
        """
        raise NotImplementedError()

    def next_n_items(self, n):
        """
        Returns next N generated item.
        :param n: Number of items to be generated.
        """
        to_ret = []
        for _ in range(n):
            to_ret.append(self.next_item())
        return to_ret

    # endregion
