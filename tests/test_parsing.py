"""
tests of numpy functions, especially searchsorted
"""
from bisect import bisect
import numpy as np

from archaic import utils, parsing


"""
utilities for generating biologically plausible recombination maps etc
"""


_default_r_bins = np.logspace(-6, -2, 17)
# translate into cM
_default_bins = utils.map_function(_default_r_bins)
_extended_r_bins = np.concatenate(([0], _default_bins, [0.49]))
_extended_bins = utils.map_function(_extended_r_bins)


def get_constant_recombination_map(L, r):
    #
    rmap = np.arange(0, r * L, r)
    return rmap


def get_random_rmap(L, lower=0, upper=5e-6):
    #
    rates = np.random.uniform(lower, upper, size=L)
    rmap = np.cumsum(rates) - rates[0]
    return rmap


"""
'naive' e.g. non-vectorized functions for counting r-binned site pairs
"""


def naively_count_site_pairs(r_map, bins, left_bound=None):
    """
    given a recombination map, bin each pair of mapped sites by looping over
    the map.

    r_map and bins must have the same linear scale; typically this will be the
    centimorgan (cM). where it is useful to count site pairs whose left
    (lower-indexed) site is below some position, a left_bound may be applied;
    this is an index of r_map and counting ceases when it is reached.

    :param r_map: vector or list representing linear recombination map
    :param bins: vector or list of recombination-distance bin edges
    :param left_bound: optional. bounds site-counting by the left site
    :return: np array of shape (len(bins) - 1) holding site pair counts
    """
    if not left_bound:
        left_bound = len(r_map) - 1

    n_bins = len(bins) - 1
    counts = np.zeros(len(bins) - 1, dtype=int)

    for i in range(left_bound):
        for j in range(i + 1, len(r_map)):
            distance = r_map[j] - r_map[i]
            bin_idx = bisect(bins, distance) - 1
            if bin_idx < n_bins:
                if bin_idx >= 0:
                    counts[bin_idx] += 1

    return counts


def naively_count_weighted_site_pairs(weights, r_map, bins, left_bound=None):
    """
    given a recombination map and site weights, loop over every pair of sites
    and bin the sums of pairwise site-weight products by distance

    :param weights: the weights to be applied to sites
    :param r_map: a linear recombination map
    :param bins: edges for recombination-distance bins
    :param left_bound: optional. bounds site-counting by the left site
    :return: np array of shape (len(bins) - 1) holding summed site-weight
        products
    """
    if len(weights) != len(r_map):
        raise ValueError('weights and r_map have mismatched lengths')

    if not left_bound:
        left_bound = len(r_map) - 1

    n_bins = len(bins) - 1
    counts = np.zeros(n_bins, dtype=float)

    for i in range(left_bound):
        for j in range(i + 1, len(r_map)):
            distance = r_map[j] - r_map[i]
            bin_idx = bisect(bins, distance) - 1
            if bin_idx < n_bins:
                if bin_idx >= 0:
                    counts[bin_idx] += weights[i] * weights[j]

    return counts


def naively_count_site_pair_array(r_map, bins, left_bound=None):
    """
    return an array holding

    :param r_map: a linear recombination map
    :param bins: recombination distance bin edges
    :param left_bound: optional. bounds site-counting by the left site
    :return: np array of shape (len(r_map), len(bins) - 1) holding per-site
        binned site-pair counts
    """
    if not left_bound:
        left_bound = len(r_map) - 1

    n_bins = len(bins) - 1
    arr = np.zeros((left_bound, n_bins), dtype=int)

    for i in range(left_bound):
        for j in range(i + 1, len(r_map)):
            distance = r_map[j] - r_map[i]
            bin_idx = bisect(bins, distance) - 1
            if bin_idx < n_bins:
                if bin_idx >= 0:
                    arr[i, bin_idx] += 1

    return arr


def naively_count_weighted_site_pair_array(
    weights,
    r_map,
    bins,
    left_bound=None
):
    """
    return an array holding sums of site-weight products binned by
    recombination distance

    :param weights: vector of site weights
    :param r_map: a linear recombination map
    :param bins: recombination distance bin edges
    :param left_bound: optional. bounds site-counting by the left site
    :return: np array of shape (len(r_map), len(bins) - 1) holding per-site
        binned site-pair counts
    """
    if len(weights) != len(r_map):
        raise ValueError('weights and r_map have mismatched lengths')

    if not left_bound:
        left_bound = len(r_map) - 1

    n_bins = len(bins) - 1
    arr = np.zeros((left_bound, n_bins), dtype=int)

    for i in range(left_bound):
        for j in range(i + 1, len(r_map)):
            distance = r_map[j] - r_map[i]
            bin_idx = bisect(bins, distance) - 1
            if bin_idx < n_bins:
                if bin_idx >= 0:
                    arr[i, bin_idx] += weights[i] * weights[j]

    return arr


def naively_count_two_sample_H2(gts_x, gts_y, r_map, bins, left_bound=None):
    #
    def get_haplotype_probs(site1, site2):
        # compute probability of sampling haplotypes 00, 10, 01, 11 at two
        # biallelic sites for 2 chromosomes
        # site1, site2 are np arrays of shape (2)
        a = np.count_nonzero(np.asanyarray(site1) == 0) / 2
        A = 1 - a
        b = np.count_nonzero(np.asanyarray(site2) == 0) / 2
        B = 1 - b
        probs = np.array([a * b, A * b, a * B, A * B])
        return probs

    def get_site_H2(site1x, site2x, site1y, site2y):
        # compute site H2 from the sum of products of haplotype probs;
        # 00 * 11 + 10 * 01 + 01 * 10 + 11 * 00
        probs_x = get_haplotype_probs(site1x, site2x)
        probs_y = get_haplotype_probs(site1y, site2y)
        site_H2 = np.prod(probs_x, np.flip(probs_y))
        return site_H2

    if len(gts_x) != len(gts_y):
        raise ValueError('genotypes have mismatched lengths')

    if len(gts_x) != len(r_map):
        raise ValueError('genotypes and r_map have mismatched lengths')

    if not left_bound:
        left_bound = len(r_map) - 1

    n_bins = len(bins) - 1
    counts = np.zeros(n_bins, dtype=float)

    for i in range(left_bound):
        for j in range(i + 1, len(r_map)):
            distance = r_map[j] - r_map[i]
            bin_idx = bisect(bins, distance) - 1
            if bin_idx < n_bins:
                if bin_idx >= 0:
                    counts[bin_idx] += get_site_H2(
                        gts_x[i], gts_x[j], gts_y[i], gts_y[j]
                    )

    return counts


"""
site pair-counting tests
"""


def test_site_pair_counting():
    #
    def sub_test(rmap, bins, left_bound):
        # assert that naive and vectorized functions match up
        naive = naively_count_site_pairs(rmap, bins, left_bound=left_bound)
        vec = parsing.count_site_pairs(rmap, bins, left_bound=left_bound)
        assert np.all(naive == vec)

    short_rmap = get_random_rmap(2_000, upper=5e-6)
    long_rmap = get_random_rmap(2_000, upper=0.002)

    sub_test(short_rmap, _default_bins, None)
    sub_test(short_rmap, _extended_bins, None)
    sub_test(short_rmap, _default_bins, 500)
    sub_test(short_rmap, _default_bins, 1500)
    sub_test(short_rmap, _extended_bins, 500)

    # rmap length exceeds largest bin edge
    sub_test(long_rmap, _default_bins, None)
    sub_test(long_rmap, _extended_bins, None)
    sub_test(long_rmap, _default_bins, 500)
    sub_test(long_rmap, _extended_bins, 1700)

    return


def test_weighted_site_pair_counting():
    #

    # the cumsum function produces some numerical inaccuracy
    # this is the tolerance for the residual between naive and vectorized
    thresh = 1e-6

    def weighting_test(rmap, bins, left_bound):
        # test match between naive unweighted parsing and weighting with 1s
        ones = np.ones(len(rmap))
        unweighted = naively_count_site_pairs(
            rmap, bins, left_bound=left_bound
        )
        weighted = parsing.count_weighted_site_pairs(
            ones, rmap, bins, left_bound=left_bound
        )
        assert np.all(unweighted == weighted)

    def sub_test(weights, rmap, bins, left_bound):
        #
        naive = naively_count_weighted_site_pairs(
            weights, rmap, bins, left_bound=left_bound
        )
        vec = parsing.count_weighted_site_pairs(
            weights, rmap, bins, left_bound=left_bound
        )
        assert np.all(
            np.logical_and(
                vec <= naive * (1 + thresh), vec >= naive * (1 - thresh)
            )
        )

    # weighting with a vector of 1s should equal unweighted result
    rmap = get_random_rmap(2000, upper=5e-6)
    long_rmap = get_random_rmap(2000, upper=0.002)
    weights = np.random.uniform(size=2000)
    integer_weights = np.random.randint(0, 2, size=2000)

    # validate weighting by comparing weighting with ones to unweighted count
    weighting_test(rmap, _default_bins, None)
    weighting_test(rmap, _extended_bins, None)
    weighting_test(rmap, _default_bins, 1250)
    weighting_test(rmap, _extended_bins, 1250)

    # naive vs vectorized weighting methods
    sub_test(weights, rmap, _default_bins, None)
    sub_test(weights, rmap, _extended_bins, None)
    sub_test(weights, rmap, _default_bins, 750)
    sub_test(weights, rmap, _extended_bins, 750)

    sub_test(weights, long_rmap, _default_bins, None)
    sub_test(weights, long_rmap, _extended_bins, None)
    sub_test(weights, long_rmap, _default_bins, 750)
    sub_test(weights, long_rmap, _extended_bins, 750)

    sub_test(integer_weights, rmap, _default_bins, None)
    sub_test(integer_weights, long_rmap, _default_bins, None)
    sub_test(integer_weights, long_rmap, _extended_bins, None)
    sub_test(integer_weights, long_rmap, _default_bins, 1000)
    sub_test(integer_weights, long_rmap, _extended_bins, 1000)

    return




"""
more rudimentary tests 
"""


def haplotype_prob_estimation():
    """
    computing two-sample H2; using products of locus-difference probabilities
    as proxies for haplotype sampling probabilities
    """
    def haplotype_prs(gts):
        # compute probability of sampling possible haplotypes from two
        # unphased sites
        # genotypes has shape (2, 2). haplotypes are 00 01 10 11
        a = (gts[0] == 0).sum() / 2
        A = (gts[0] == 1).sum() / 2
        b = (gts[1] == 0).sum() / 2
        B = (gts[1] == 1).sum() / 2
        prs = np.array([a * b, a * B, A * b, A * B])
        return prs

    def haplotype_H2(gts_x, gts_y):
        #
        return (haplotype_prs(gts_x) * np.flip(haplotype_prs(gts_y))).sum()

    def two_sample_H(gt_x, gt_y):
        # compute probabilities of sampling a distinct allele from x and y at
        # each locus
        _gt_x = gt_x[:, :, np.newaxis]
        _gt_y = gt_y[:, np.newaxis]
        site_H = (_gt_x != _gt_y).sum((2, 1)) / 4
        return site_H

    def sitewise_H2(gts_x, gts_y):
        #
        H_x = two_sample_H(gts_x, gts_y)
        return H_x[0] * H_x[1]

    def get_all_gts():
        # enumerate all the two-locus biallelic genotypes possible
        gts = []
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    for l in [0, 1]:
                        gts.append(np.array([[i, j], [k, l]]))
        return gts

    gts = get_all_gts()
    for i in range(16):
        for j in range(i, 16):
            hap_H2 = haplotype_H2(gts[i], gts[j])
            site_H2 = sitewise_H2(gts[i], gts[j])
            print(hap_H2, site_H2)
            assert hap_H2 == site_H2
    return 0

