"""
tests of numpy functions, especially searchsorted
"""
import numpy as np


def subset_with_searchsorted():
    # searchsorting a subset of a vector of integers on that vector gives
    # the indices of subset elements
    positions = np.arange(1, 1001)
    subset_positions = np.linspace(10, 900, 90, dtype=int)
    idx = np.searchsorted(positions, subset_positions)
    assert np.all(positions[idx] == subset_positions)
    return 0


