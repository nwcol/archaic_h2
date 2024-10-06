"""
houses the site pair counting functions that underlie the estimation of the
two-locus heterozygosity
"""
from bisect import bisect
import numpy as np
import numpy.ma as ma

from archaic import util
















def _count_site_pairs(
    positions,
    rcoords,
    rmap,
    bins,
    left_bound=None
):
    # uses interpolation rather than searchsorting. currently WIP and not a
    # high priority
    if len(rcoords) != len(rmap):
        raise ValueError('rcoords length mismatches rmap')
    if left_bound is None:
        left_bound = len(positions)
    # interpolate r-map values
    site_r = np.interp(
        positions,
        rcoords,
        rmap,
        left=rmap[0],
        right=rmap[-1]
    )
    coords_in_pos_idx = np.searchsorted(positions, rcoords)
    coords_in_pos_idx[coords_in_pos_idx == len(positions)] -= 1
    edges = np.zeros(len(bins), dtype=int)

    for i, b in enumerate(bins):
        edges[i] = np.floor(
            np.interp(
                site_r[:left_bound] + b,
                rmap,
                coords_in_pos_idx,
                left=0,
                right=len(positions) - 1
            )
        ).sum()

    counts = np.diff(edges)

    return counts


def _count_weighted_site_pairs(
    positions,
    rcoords,
    rmap,
    bins,
    weights,
    left_bound=None
):
    # uses interpolation rather than searchsorting. currently WIP and not a
    # high priority
    # currently inaccurate
    if len(rcoords) != len(rmap):
        raise ValueError('rcoords length mismatches rmap')
    if left_bound is None:
        left_bound = len(positions)

    site_r = np.interp(
        positions,
        rcoords,
        rmap,
        left=rmap[0],
        right=rmap[-1]
    )
    coords_in_pos_idx = np.searchsorted(positions, rcoords)
    cum_weights = np.cumsum(weights)
    weighted_edges = np.zeros(len(bins), dtype=float)

    for i, b in enumerate(bins):
        idx = np.ceil(
            np.interp(
                site_r[:left_bound] + b,
                rmap,
                coords_in_pos_idx,
                left=1,
                right=len(positions)
            )
        ).astype(int) - 1
        weighted_edges[i] = np.dot(weights, cum_weights[idx])

    weighted_counts = np.diff(weighted_edges)

    return weighted_counts


def count_site_pairs(
    r_map,
    bins,
    left_bound=None
):
    """
    compute numbers of site pairs, binned by recombination distances

    :param r_map: recombination map
    :param bins: recombination distance bins. the unit/scale must match that
        of r_map (typically this will be centiMorgans)
    :param left_bound: optional. if provided, specifies the highest index
        allowed for the left (lower-index) site in pair-counting
    :return: vector of pair counts
    """
    if len(r_map) < 2:
        return np.zeros(len(bins) - 1)

    if not left_bound:
        left_bound = len(r_map)
    else:
        if left_bound > len(r_map):
            raise ValueError('left_bound index exceeds map length')

    # this is a LARGE operation and may consume very large amounts of memory
    site_bins = r_map[:left_bound, np.newaxis] + bins[np.newaxis, :]
    bin_edges = np.searchsorted(r_map, site_bins)

    # this adjustment prevents pair over-counting and self-pairing
    over_counts = bin_edges[:, 0] <= np.arange(left_bound)
    bin_edges[over_counts, 0] = np.arange(left_bound)[over_counts] + 1
    num_pairs = np.diff(bin_edges.sum(0))
    print(
        util.get_time(),
        f'site pair counts computed for {left_bound} loci'
    )
    return num_pairs


def count_weighted_site_pairs(
    weights,
    r_map,
    bins,
    left_bound=None,
    verbosity=1e6
):
    """
    compute numbers of site pairs binned by recombination distances, counting
    each pair as the product of site weights

    :param weights: weights associated with each site
    :param r_map: recombination map
    :param bins: recombination bins. unit must match r_map
    :param left_bound: optional. if provided, specifies the highest index
        allowed for the left (lower-index) site in pair-counting
    :param verbosity: status printout interval
    :return: vector of weighted pair counts
    """
    if len(weights) != len(r_map):
        raise ValueError('weights and r_map have mismatched lengths')

    if len(r_map) < 2:
        return np.zeros(len(bins) - 1)

    if not left_bound:
        left_bound = len(r_map)
    else:
        if left_bound > len(r_map):
            raise ValueError('left_bound index exceeds map length')

    site_bins = r_map[:left_bound, np.newaxis] + bins[np.newaxis, :]
    bin_edges = np.searchsorted(r_map, site_bins)
    # decrementing by 1 gives the proper indices for cum_weights below
    bin_edges -= 1

    # correction for pair over-counting
    over_counts = bin_edges[:, 0] < np.arange(left_bound)
    bin_edges[over_counts, 0] = np.arange(left_bound)[over_counts]

    # for a site, we compute weighted counts by taking the product of the site
    # weight with the difference in cumulative weight counts at bin edges
    cum_weights = np.cumsum(weights)

    num_pairs = np.zeros(len(bins) - 1, dtype=float)

    for i in np.arange(left_bound):
        if weights[i] > 0:
            num_pairs += weights[i] * np.diff(cum_weights[bin_edges[i]])
            if i % verbosity == 0:
                if i > 0:
                    print(
                        util.get_time(),
                        f'weighted site pairs counted at site {i}'
                    )
    print(
        util.get_time(),
        f'weighted site pair counts computed for {left_bound} loci'
    )
    return num_pairs


"""
mutation-rate weighted pair counting functions
"""


def compute_bin_averaged_u_weight(
    positions,
    u_map,
    r_map,
    bins,
    windows,
    bounds
):
    # compute the denominator for a weighted H2 statistic
    # for this statistic we use as the bin denominator
    # sum(u_l * u_r)  / [mean(u_l) * mean(u_r)], where sums/means are within
    # the bin

    denom = np.zeros((len(windows), len(bins) - 1), dtype=float)

    for w, (window, bound) in enumerate(zip(windows, bounds)):
        start = np.searchsorted(positions, window[0])
        r_end = np.searchsorted(positions, bounds[w])
        l_end = np.searchsorted(positions[start:], window[1])

        bin_u_prods = compute_binned_u_prods(
            u_map[start:r_end],
            r_map[start:r_end],
            bins,
            l_lim=l_end
        )
        num_pairs, sum_l, sum_r = compute_binned_u_sums(
            u_map[start:r_end],
            r_map[start:r_end],
            bins,
            l_lim=l_end
        )
        mean_l = np.divide(sum_l, num_pairs, where=num_pairs > 0)
        mean_r = np.divide(sum_r, num_pairs, where=num_pairs > 0)
        prod_rl = mean_l * mean_r
        denom[w] = np.divide(bin_u_prods, prod_rl, where=prod_rl > 0)

    return denom


def compute_chrom_averaged_u_weight(
    positions,
    u_map,
    r_map,
    bins,
    windows
):
    # compute the denominator for a weighted H2 statistic
    # here, the denominator for a bin equals the sum of mutation rate pair
    # products over the mean mutation rate pair product across all bins
    # e.g. sum(u_l * u_r)(bin) / mean(u_l * u_r)(tot)

    # this array has a row for each window
    bin_u_prods = np.zeros((len(windows), len(bins) - 1), dtype=float)
    num_pairs = 0

    for w, (w_start, w_l_end, w_r_end) in enumerate(windows):
        start = np.searchsorted(positions, w_start)
        r_end = np.searchsorted(positions, w_r_end)
        l_end = np.searchsorted(positions[start:], w_l_end)

        #
        bin_u_prods[w] = compute_binned_u_prods(
            u_map[start:r_end],
            r_map[start:r_end],
            bins,
            l_lim=l_end
        )
        num_pairs += count_site_pairs(
            r_map[start:r_end],
            np.array([bins[0], bins[-1]]),
            left_bound=l_end
        ).sum()

    tot_ul_ur = bin_u_prods.sum()
    print(num_pairs)
    mean_ul_ur = tot_ul_ur / num_pairs
    print(mean_ul_ur)
    denom = bin_u_prods / mean_ul_ur
    return denom


def chrom_weighted_u2(
    positions,
    u_map,
    r_map,
    bins,
    windows,
    bounds
):
    # compute the denominator for a weighted H2 statistic
    # here, the denominator for a bin equals the sum of mutation rate pair
    # products over the square of the average mutation rate.
    # this is some sort of approximation to the average product of mutation
    # rates for sites which are actually counted
    # e.g. sum(u_l * u_r)(bin) / mean(u) ** 2
    bin_u_prods = np.zeros((len(windows), len(bins) - 1), dtype=float)

    for w, (window, bound) in enumerate(zip(windows, bounds)):
        start = np.searchsorted(positions, window[0])
        r_end = np.searchsorted(positions, bounds[w])
        l_end = np.searchsorted(positions[start:], window[1])

        bin_u_prods[w] = compute_binned_u_prods(
            u_map[start:r_end],
            r_map[start:r_end],
            bins,
            l_lim=l_end
        )

    u_squared = np.mean(u_map) ** 2
    denom = bin_u_prods / u_squared
    return denom


def compute_mean_prod(u_map, bound=None):
    # NOT IN USE
    # compute the average product u_left * u_right across all site pairs
    if bound is None:
        u_map_0 = u_map
        u_map_1 = None
    else:
        if bound < len(u_map):
            u_map_0 = u_map[:bound]
            u_map_1 = u_map[bound:]
            print(len(u_map_0), len(u_map_1))
        else:
            u_map_0 = u_map
            u_map_1 = None

    n_sites_0 = len(u_map_0)
    cum_u_0 = np.cumsum(u_map_0)
    sum_prods = 0
    num_pairs = 0

    for i, u in enumerate(u_map_0):
        sum_prods += u * (cum_u_0[-1] - cum_u_0[i])
        num_pairs += n_sites_0 - i - 1

    if u_map_1 is not None:
        n_sites_1 = len(u_map_1)
        cum_u_1 = np.cumsum(u_map_1)

        for i, u in enumerate(u_map_1):
            sum_prods += u * (cum_u_1[-1] - cum_u_1[i])
            num_pairs += n_sites_1 - i - 1

    mean_prod = sum_prods / num_pairs
    return mean_prod


def compute_binned_u_prods(
    u_map,
    r_map,
    bins,
    l_lim=None,
    verbosity=1e6
):
    # compute sums of products of left * right locus mutation rates
    if not l_lim:
        l_lim = len(r_map)

    site_bins = r_map[:l_lim, np.newaxis] + bins[np.newaxis, :]
    bin_edges = np.searchsorted(r_map, site_bins)
    bin_edges -= 1

    # correction
    over_counts = bin_edges[:, 0] < np.arange(l_lim)
    bin_edges[over_counts, 0] = np.arange(l_lim)[over_counts]

    cum_u = np.cumsum(u_map)

    sum_lr = np.zeros(len(bins) - 1, dtype=float)

    for i, l_u in enumerate(u_map[:l_lim]):
        if l_u > 0:
            sum_lr += l_u * np.diff(cum_u[bin_edges[i]])

            if i % verbosity == 0:
                if i > 0:
                    print(
                        util.get_time(),
                        f'weighted site pairs counted at site {i}'
                    )

    return sum_lr


def compute_binned_u_sums(
    u_map,
    r_map,
    bins,
    l_lim=None,
    verbosity=1e6
):
    # compute sums of left and right locus mutation rates
    if not l_lim:
        l_lim = len(r_map)

    site_bins = r_map[:l_lim, np.newaxis] + bins[np.newaxis, :]
    bin_edges = np.searchsorted(r_map, site_bins)
    bin_edges -= 1

    # correction to make minimum r index equal i + 1
    over_counts = bin_edges[:, 0] < np.arange(l_lim)
    bin_edges[over_counts, 0] = np.arange(l_lim)[over_counts]

    cum_u = np.cumsum(u_map)

    sum_l = np.zeros(len(bins) - 1, dtype=float)
    sum_r = np.zeros(len(bins) - 1, dtype=float)
    num_pairs = np.zeros(len(bins) - 1, dtype=int)

    for i, l_u in enumerate(u_map[:l_lim]):
        if l_u > 0:
            n_rs = np.diff(bin_edges[i])
            num_pairs += n_rs
            sum_l += l_u * n_rs
            sum_r += np.diff(cum_u[bin_edges[i]])

            if i % verbosity == 0:
                if i > 0:
                    print(
                        util.get_time(),
                        f'weighted site pairs counted at site {i}'
                    )

    return num_pairs, sum_l, sum_r
