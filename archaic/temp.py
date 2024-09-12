# development of a new weighted H2 statistic
from bisect import bisect
import numpy as np

from archaic import util


def sum_prod(x):

    s = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            s += x[i] * x[j]
    return s



def loop_func(
    u_map,
    r_map,
    bins,
    max_left=None
):
    # compute the sum over terms u_a / mean u_a * u_b / mean u_b
    if not max_left:
        max_left = len(r_map)

    n_bins = len(bins) - 1

    n_pairs = np.zeros(n_bins)
    sum_u_a = np.zeros(n_bins, dtype=float)
    sum_u_b = np.zeros(n_bins, dtype=float)
    sum_prod_ab = np.zeros(n_bins, dtype=float)

    for i in range(max_left):
        for j in range(i + 1, len(r_map)):
            k = bisect(bins, r_map[j] - r_map[i]) - 1
            if k >= 0:
                if k < n_bins:
                    n_pairs[k] += 1
                    sum_prod_ab[k] += u_map[i] * u_map[j]
                    sum_u_a[k] += u_map[i]
                    sum_u_b[k] += u_map[j]

    nonzero = n_pairs > 0
    mean_u_a = np.divide(sum_u_a, n_pairs, where=nonzero)
    mean_u_b = np.divide(sum_u_b, n_pairs, where=nonzero)
    prod_means = np.multiply(mean_u_a, mean_u_b)
    terms = np.divide(sum_prod_ab, prod_means, where=nonzero)
    terms[terms < 1e-100] = 0
    return terms


def vec_func(u_map, r_map, bins, l_lim=None, verbosity=1e6):
    #
    if not l_lim:
        l_lim = len(r_map)

    site_bins = r_map[:l_lim, np.newaxis] + bins[np.newaxis, :]
    bin_edges = np.searchsorted(r_map, site_bins)
    bin_edges -= 1

    # compute the numerator sum u_l * u_r
    cum_u = np.cumsum(u_map)
    sum_prod_lr = np.zeros(len(bins) - 1, dtype=float)
    sum_l = np.zeros(len(bins) - 1, dtype=float)
    sum_r = np.zeros(len(bins) - 1, dtype=float)
    num_pairs = np.zeros(len(bins) - 1, dtype=float)

    for i in np.arange(l_lim):
        if u_map[i] > 0:
            r_u = np.diff(cum_u[bin_edges[i]])
            num_pairs_i = np.diff(bin_edges[i])

            sum_prod_lr += u_map[i] * r_u
            num_pairs += num_pairs_i
            sum_l += u_map[i] * num_pairs_i
            sum_r += r_u

            if i % verbosity == 0:
                if i > 0:
                    print(
                        util.get_time(),
                        f'weighted site pairs counted at site {i}'
                    )

    nonzero = num_pairs > 0
    mean_l = np.divide(sum_l, num_pairs, where=nonzero)
    mean_r = np.divide(sum_r, num_pairs, where=nonzero)
    prod_means = np.multiply(mean_l, mean_r)
    terms = np.divide(sum_prod_lr, prod_means, where=nonzero)
    terms[terms < 1e-100] = 0
    return terms
