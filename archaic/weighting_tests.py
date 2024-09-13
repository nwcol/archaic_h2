# a couple of different methods for weighting H2 by the local mutation rate
import numpy as np
import time

from archaic import util, parsing
from archaic.parsing import _default_bins


def parse_weighted_H2(
    mask_fname,
    vcf_fname,
    map_fname,
    weight_fname,
    bins=None,
    windows=None,
    bounds=None,
    inverse_weight=False,
    c_func=None
):
    #
    if isinstance(bins, np.ndarray):
        pass
    elif isinstance(bins, str):
        bins = np.loadtxt(bins, dtype=float)
    else:
        print(util.get_time(), 'parsing with default bins')
        bins = _default_bins

    if isinstance(windows, np.ndarray):
        if windows.ndim == 1:
            windows = windows[np.newaxis]
        if windows.shape[1] == 3:
            bounds = windows[:, 2]
            windows = windows[:, :2]
    elif isinstance(windows, str):
        window_arr = np.loadtxt(windows, dtype=int)
        if window_arr.ndim == 1:
            window_arr = window_arr[np.newaxis]
        windows = window_arr[:, :2]
        bounds = window_arr[:, 2]
    else:
        windows = None

    t0 = time.time()
    mask_regions = util.read_mask_file(mask_fname)
    mask_positions = util.get_mask_positions(mask_regions)
    sample_ids, vcf_positions, genotype_arr = \
        util.read_vcf_genotypes(vcf_fname, mask_regions)
    r_map = util.read_map_file(map_fname, mask_positions)
    print(util.get_time(), "files loaded")

    # load weight file
    # this could be improved
    if weight_fname.endswith('.npz'):
        weight_file = np.load(weight_fname)
        weight_positions = weight_file['positions']
        all_weights = weight_file['rates']
        weights = all_weights[np.searchsorted(weight_positions, mask_positions)]
    else:
        regions, data = util.read_bedgraph(weight_fname)
        # assign a mutation rate to each point
        idx = np.searchsorted(regions[:, 1], mask_positions)
        weights = data['u'][idx]
    if inverse_weight:
        weights = weights ** -1
    print(util.get_time(), 'weight files loaded')

    # one-locus H doesn't get weighted
    num_sites, num_H = parsing.compute_H(
        mask_positions,
        genotype_arr,
        vcf_positions,
        windows=windows
    )
    print(util.get_time(), 'one-locus H computed')

    # two-locus H (H2)
    num_pairs, num_H2 = compute_weighted_H2(
        mask_positions,
        genotype_arr,
        vcf_positions,
        r_map,
        weights,
        bins=bins,
        windows=windows,
        bounds=bounds,
        c_func=c_func
    )
    print(util.get_time(), 'two-locus H computed')

    n = len(sample_ids)
    stat_ids = [
        (sample_ids[i], sample_ids[j])
        for i in np.arange(n) for j in np.arange(i, n)
    ]
    ret = dict(
        n_sites=num_sites,
        H_counts=num_H,
        n_site_pairs=num_pairs,
        H2_counts=num_H2,
        r_bins=bins,
        ids=stat_ids,
        windows=windows,
        bounds=bounds
    )

    t = np.round(time.time() - t0, 0)
    chrom_num = util.read_vcf_contig(vcf_fname)
    print(
        util.get_time(),
        f'{len(mask_positions)} sites on '
        f'chromosome {chrom_num} parsed in\t{t} s'
    )
    return ret



def compute_weighted_H2(
    positions,
    genotype_arr,
    genotype_pos,
    r_map,
    u_map,
    bins=None,
    windows=None,
    bounds=None,
    c_func=None
):

    bins = util.map_function(bins)

    vcf_r_map = r_map[np.searchsorted(positions, genotype_pos)]
    _, n_samples, __ = genotype_arr.shape
    n_stats = util.n_choose_2(n_samples) + n_samples
    num_H2 = np.zeros((len(windows), n_stats, len(bins) - 1))

    c_funcs = {
        'bin_weighted': bin_weighted,
        'chrom_weighted_uu': chrom_weighted_ulur,
        'chrom_weighted_u2': chrom_weighted_u2
    }

    denom = c_funcs[c_func](
        positions,
        u_map,
        r_map,
        bins,
        windows,
        bounds
    )

    for z, (window, bound) in enumerate(zip(windows, bounds)):
        vcf_start = np.searchsorted(genotype_pos, window[0])
        vcf_rbound = np.searchsorted(genotype_pos, bound)
        vcf_lbound = np.searchsorted(genotype_pos[vcf_start:], window[1])

        win_vcf_positions = genotype_pos[vcf_start:vcf_rbound]
        win_vcf_r_map = vcf_r_map[vcf_start:vcf_rbound]
        win_genotype_arr = genotype_arr[vcf_start:vcf_rbound]

        k = 0

        for i in range(n_samples):
            for j in range(i, n_samples):
                if i == j:
                    gts = win_genotype_arr[:, i]
                    site_H = gts[:, 0] != gts[:, 1]
                    sample_lbound = np.searchsorted(
                        win_vcf_positions[site_H], window[1]
                    )
                    num_H2[z, k] = parsing.count_site_pairs(
                        win_vcf_r_map[site_H],
                        bins,
                        left_bound=sample_lbound
                    )
                else:
                    gts_i = win_genotype_arr[:, i]
                    gts_j = win_genotype_arr[:, j]
                    site_H = parsing.get_two_sample_site_H(gts_i, gts_j)
                    num_H2[z, k] = parsing.count_weighted_site_pairs(
                        site_H,
                        win_vcf_r_map,
                        bins,
                        left_bound=vcf_lbound
                    )
                k += 1

        print(util.get_time(), f'computed H2 in window {z}')

    return denom, num_H2


def bin_weighted(
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

        bin_sum_ulur = compute_binned_ulur(
            u_map[start:r_end],
            r_map[start:r_end],
            bins,
            l_lim=l_end
        )
        num_pairs, sum_l, sum_r = compute_binned_ul_ur(
            u_map[start:r_end],
            r_map[start:r_end],
            bins,
            l_lim=l_end
        )
        mean_l = np.divide(sum_l, num_pairs, where=num_pairs > 0)
        mean_r = np.divide(sum_r, num_pairs, where=num_pairs > 0)
        prod_rl = mean_l * mean_r
        denom[w] = np.divide(bin_sum_ulur, prod_rl, where=prod_rl > 0)

    return denom


def chrom_weighted_ulur(
    positions,
    u_map,
    r_map,
    bins,
    windows,
    bounds
):
    # compute the denominator for a weighted H2 statistic
    # here, the denominator for a bin equals the sum of mutation rate pair
    # products over the mean mutation rate pair product across all bins
    # e.g. sum(u_l * u_r)(bin) / mean(u_l * u_r)(tot)

    # this array has a row for each window
    bin_sum_ul_ur = np.zeros((len(windows), len(bins) - 1), dtype=float)
    num_pairs = 0

    for w, (window, bound) in enumerate(zip(windows, bounds)):
        start = np.searchsorted(positions, window[0])
        r_end = np.searchsorted(positions, bounds[w])
        l_end = np.searchsorted(positions[start:], window[1])

        #
        bin_sum_ul_ur[w] = compute_binned_ulur(
            u_map[start:r_end],
            r_map[start:r_end],
            bins,
            l_lim=l_end
        )
        num_pairs += parsing.count_site_pairs(
            r_map[start:r_end],
            np.array([bins[0], bins[-1]]),
            left_bound=l_end
        ).sum()

    tot_ul_ur = bin_sum_ul_ur.sum()
    print(num_pairs)
    mean_ul_ur = tot_ul_ur / num_pairs
    print(mean_ul_ur)
    denom = bin_sum_ul_ur / mean_ul_ur
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
    bin_sum_ul_ur = np.zeros((len(windows), len(bins) - 1), dtype=float)

    for w, (window, bound) in enumerate(zip(windows, bounds)):
        start = np.searchsorted(positions, window[0])
        r_end = np.searchsorted(positions, bounds[w])
        l_end = np.searchsorted(positions[start:], window[1])

        bin_sum_ul_ur[w] = compute_binned_ulur(
            u_map[start:r_end],
            r_map[start:r_end],
            bins,
            l_lim=l_end
        )

    u_squared = np.mean(u_map) ** 2
    denom = bin_sum_ul_ur / u_squared
    return denom


def compute_mean_prod(u_map, bound=None):
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


def compute_binned_ulur(
    u_map,
    r_map,
    bins,
    l_lim=None,
    verbosity=1e6
):
    # computes sum of products
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


def compute_binned_ul_ur(
    u_map,
    r_map,
    bins,
    l_lim=None,
    verbosity=1e6
):
    # computes separate sums for left and right sites
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
