"""
Various utilities that are widely used
"""
from datetime import datetime
import numpy as np


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


def get_pair_names(items):

    pairs = get_pairs(items)
    pair_names = [f"{x},{y}" for x, y in pairs]
    return pair_names


def get_pair_idxs(n):
    # return a list of 2-tuples containing pairs of indices up to n
    pairs = []
    for i in np.arange(n):
        for j in np.arange(i + 1, n):
            pairs.append((i, j))
    return pairs


"""
Combinatorics
"""


def n_choose_2(n):
    #
    return int(n * (n - 1) * 0.5)


def n_choose_m(n, m):
    # implement
    return None


"""
Stats printouts
"""


def get_time():
    # return a string giving the date and time
    return "[" + datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S") + "]"
