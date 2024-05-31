
"""
Random utilities that are widely used
"""

from datetime import datetime
import numpy as np


"""

    pair_names = [f"{x},{y}" for (x, y) in sample_pairs]
"""



"""
Indexing
"""


def get_pairs(items):
    # return a list of 2-tuples containing every pair in 'items'
    n = len(items)
    pairs = []
    for i in np.arange(n):
        for j in np.arange(i + 1, n):
            pair = [items[i], items[j]]
            pair.sort()
            pairs.append((pair[0], pair[1]))
    return pairs


def get_pair_idxs(n):
    # return a list of 2-tuples contaning pairs of indices up to n
    pairs = []
    for i in np.arange(n):
        for j in np.arange(i + 1, n):
            pairs.append((i, j))
    return pairs


"""
Printing stuff
"""


def get_time():
    # get a string giving the date and time
    return "[" + datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S") + "]"



