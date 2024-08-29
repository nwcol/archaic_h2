"""
tests of numpy functions, especially searchsorted
"""
from bisect import bisect
import numpy as np

from archaic import utils, parsing


"""
utilities for generating biologically plausible recombination maps etc
"""


_default_bins = np.logspace(-6, -2, 17)
_extended_bins = np.concatenate(([0], _default_bins, [0.499]))


def get_constant_recombination_map(r, L, n_sites=None):
    #
    if n_sites is None:
        r_map = r * np.arange(1, L + 1)
    else:
        if n_sites > L:
            raise ValueError('f')
        positions = np.random.randint(1, L + 1, )
    return r_map


def get_random_recombination_map(length, lower=0, upper=5e-8):
    #
    rates = np.random.uniform(lower, upper, size=length)
    return np.cumsum(rates)


def get_random_weights():


    return 0


def get_random_integer_weights(length):
    #


    return 0


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
        left_bound = len(r_map)

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
    if not left_bound:
        left_bound = len(r_map)

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
        left_bound = len(r_map)

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


"""
site pair-counting tests
"""


def test_site_pair_counting():






    return 0


def test_weighted_site_pair_counting():




    return 0



def naive_two_sample_H2_count(genotypes, r_map, bins, left_bound=None):
    #

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

    if not left_bound:
        left_bound = len(r_map)

    gt_x = genotypes[:, 0]
    gt_y = genotypes[:, 1]

    counts = np.zeros(len(bins) - 1, dtype=float)

    for i in range(left_bound):
        for j in range(i + 1, len(r_map)):
            distance = r_map[j] - r_map[i]
            idx = bisect(bins, distance) - 1
            if idx < len(bins) - 1:
                if idx > -1:
                    hap_prs_x = haplotype_prs(gt_x[[i, j]])
                    hap_prs_y = haplotype_prs(gt_y[[i, j]])
                    counts[idx] += (hap_prs_x * np.flip(hap_prs_y)).sum()

    return counts


def _test_two_sample_H2_counting():

    length = 950
    r_map = np.linspace(0.001, 101, length)
    bins = np.logspace(-6, -2, 17)
    bins0 = np.concatenate([[0], bins])
    cM_bins = utils.map_function(bins)
    cM_bins0 = utils.map_function(bins0)
    genotypes_x = np.zeros((length, 2))
    genotypes_y = np.zeros((length, 2))
    # randomly enter 1s into genotypes
    for i in range(length):
        if np.random.random() > 0.6:
            genotypes_x[i, 0] = 1
        if np.random.random() > 0.6:
            genotypes_x[i, 1] = 1
        if np.random.random() > 0.5:
            genotypes_y[i, 0] = 1
        if np.random.random() > 0.5:
            genotypes_y[i, 1] = 1

    genotypes = np.stack([genotypes_x, genotypes_y], axis=1)

    # bins don't include 0 edge
    naive = naive_two_sample_H2_count(genotypes, r_map, cM_bins)
    vec = parsing.count_two_sample_H2(genotypes, r_map, bins)
    assert np.all(naive == vec)

    # bins do include a 0 edge
    naive = naive_two_sample_H2_count(genotypes, r_map, cM_bins0)
    vec = parsing.count_two_sample_H2(genotypes, r_map, bins0)
    assert np.all(naive == vec)
    return 0


def _test_pair_counting():
    #
    r_map = np.logspace(-8, 2.1, 1456)
    bins = np.logspace(-6, -2, 17)
    cM_bins = utils.map_function(bins)
    naive = naive_pair_count(r_map, cM_bins)
    vec = parsing.count_site_pairs(r_map, bins)
    assert np.all(naive == vec)

    bound = 567
    naive = naive_pair_count(r_map, cM_bins, left_bound=bound)
    vec = parsing.count_site_pairs(r_map, bins, left_bound=bound)
    assert np.all(naive == vec)

    # self and pre-counting test
    r_map = np.array([1, 1, 1, 1, 2, 2, 3, 4, 45, 45])
    bins = np.array([0, .05, .11, .45])
    cM_bins = utils.map_function(bins)
    naive = naive_pair_count(r_map, cM_bins)
    vec = parsing.count_site_pairs(r_map, bins)
    assert np.all(naive == vec)
    return 0



def _test_scaled_pair_counting():
    # numeric precision prevents asserts from working
    r_map = np.logspace(-7, 2.1, 1456)
    bins = np.logspace(-6, -2, 17)
    scale = np.random.random(size=1456)
    cM_bins = utils.map_function(bins)
    naive = naive_scaled_pair_count(r_map, cM_bins, scale)
    vec = parsing.count_scaled_site_pairs(r_map, bins, scale)
    print(abs(naive-vec) / naive)

    naive = naive_scaled_pair_count(r_map, cM_bins, 1 / scale)
    vec = parsing.count_scaled_site_pairs(r_map, bins, 1 / scale)
    print(abs(naive - vec) / naive)

    bins_to_zero = np.concatenate([[0], bins])
    naive = naive_scaled_pair_count(bins_to_zero, cM_bins, scale)
    vec = parsing.count_scaled_site_pairs(bins_to_zero, bins, scale)
    print(abs(naive - vec) / naive)

    bound = 567
    naive = naive_scaled_pair_count(r_map, cM_bins, scale, left_bound=bound)
    vec = parsing.count_scaled_site_pairs(r_map, bins, scale, left_bound=bound)
    print(abs(naive - vec) / naive)
    return 0




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


def bin_naive(vals, bins):
    # bins are of type [ )
    counts = np.zeros(len(bins) - 1)
    for x in vals:
        bin_idx = bisect(bins, x) - 1
        if bin_idx < len(bins) - 1:
            if bin_idx > -1:
                counts[bin_idx] += 1
    return counts


def bin_with_cumsum(vals, bins):
    # 1-dimensional version of the problem
    counts = np.diff(np.searchsorted(vals, bins))
    return counts


def _test_binning():
    #
    def test(vals, bins):
        naive = bin_naive(vals, bins)
        vec = bin_with_cumsum(vals, bins)
        assert len(naive) == len(vec)
        assert np.all(naive == vec)

    test(np.arange(45), np.linspace(0, 50, 6))
    test(np.arange(45), np.linspace(5, 50, 10))
    test(np.arange(100), np.linspace(1, 50, 5))
    return 0


def map_binning():
    #
    def naive_func(val_map, bins):
        #
        idxs = np.zeros(len(val_map))
        for i, x in enumerate(val_map):
            bin_idx = bisect(bins, x) - 1
            if bin_idx < len(bins) - 1:
                if bin_idx > -1:
                    idxs[i] = bin_idx
        return idxs

    def vec_func(val_map, bins):
        #
        idxs = np.searchsorted(val_map, bins) - 1
        return idxs


def weighted_pair_counting():
    # here we weight by the right site value
    def naive_func(vals, val_map, bins):
        #
        weighted_counts = np.zeros(len(bins) - 1, dtype=float)

        for i in range(len(val_map)):
            for j in range(i + 1, len(val_map)):
                distance = val_map[j] - val_map[i]
                bin_idx = bisect(bins, distance) - 1
                if bin_idx < len(bins) - 1:
                    if bin_idx > -1:
                        weighted_counts[bin_idx] += vals[j]

        return weighted_counts

    def vec_func(vals, val_map, bins):
        # this is seemingly unworkable
        # make an array of site-centered bins
        site_bins = val_map[:, np.newaxis] + bins[np.newaxis]
        edges = np.searchsorted(val_map, site_bins)
        too_low = edges[:, 0] <= np.arange(len(val_map))
        edges[too_low, 0] = np.arange(len(val_map))[too_low] + 1
        cum_vals = np.concatenate([[0], np.cumsum(vals)])
        weighted_counts = np.diff(cum_vals[edges]).sum(0)
        return weighted_counts

    def test(vals, val_map, bins):
        #
        naive = naive_func(vals, val_map, bins)
        vec = vec_func(vals, val_map, bins)
        assert len(naive) == len(vec)
        assert np.all(naive == vec)

    test(
        np.ones(10),
        np.array([1, 1, 1, 1, 2, 3, 4, 5, 6, 10]),
        np.array([0, 5.5, 10])
    )
    test(np.ones(100), np.linspace(0, 101, 100), np.logspace(-1, 2, 6))
    test(np.ones(100), np.logspace(-2, 2, 100), np.logspace(-1, 2, 6))
    test(np.arange(100), np.logspace(-2, 2, 100), np.logspace(-1, 2, 6))
    return 0


def product_weighted_pair_counting():
    #
    def naive_func(vals, val_map, bins):
        #
        weighted_counts = np.zeros(len(bins) - 1, dtype=float)

        for i in range(len(val_map)):
            for j in range(i + 1, len(val_map)):
                distance = val_map[j] - val_map[i]
                bin_idx = bisect(bins, distance) - 1
                if bin_idx < len(bins) - 1:
                    if bin_idx > -1:
                        weighted_counts[bin_idx] += vals[i] * vals[j]

        return weighted_counts

    def vec_func(vals, val_map, bins):
        # this is seemingly unworkable
        # make an array of site-centered bins
        site_bins = val_map[:, np.newaxis] + bins[np.newaxis]
        edges = np.searchsorted(val_map, site_bins)
        too_low = edges[:, 0] <= np.arange(len(val_map))
        edges[too_low, 0] = np.arange(len(val_map))[too_low] + 1
        cum_vals = np.concatenate([[0], np.cumsum(vals)])
        weighted_counts = np.zeros(len(bins) - 1)

        for i in range(len(val_map)):
            weighted_counts += vals[i] * np.diff(cum_vals[edges[i]])

        return weighted_counts

    def test(vals, val_map, bins):
        #
        naive = naive_func(vals, val_map, bins)
        vec = vec_func(vals, val_map, bins)
        assert len(naive) == len(vec)
        assert np.all(naive == vec)

    test(
        np.ones(10),
        np.array([1, 1, 1, 1, 2, 3, 4, 5, 6, 10]),
        np.array([0, 5.5, 10])
    )
    test(np.ones(100), np.linspace(0, 101, 100), np.logspace(-1, 2, 6))
    test(np.ones(100), np.logspace(-2, 2, 100), np.logspace(-1, 2, 6))
    test(np.arange(100), np.logspace(-2, 2, 100), np.logspace(-1, 2, 6))
    return 0
