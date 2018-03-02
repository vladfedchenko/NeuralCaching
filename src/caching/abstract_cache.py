class AbstractCache:

    # region Private variables

    # endregion

    # region Protected variables

    _cache_size = 0
    _free_cache = 0

    # endregion

    # region Public variables, properties

    # endregion

    # region Constructors

    def __init__(self, size):
        self._cache_size = size
        self._free_cache = size

    # endregion

    # region Private methods

    # endregion

    # region Protected methods

    def _store_object(self, guid, size):
        raise NotImplementedError()

    def _remove_object(self, guid, size):
        raise NotImplementedError()

    # endregion

    # region Public methods

    def request_object(self, guid, size, time):
        raise NotImplementedError()

    # endregion