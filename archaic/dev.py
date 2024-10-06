
from bisect import bisect
import numpy as np
import numpy.ma as ma

from archaic import util


def _count_num_pairs(rmap, bins, llim=None):
    """
    given a recombination map, bin each pair of mapped sites by looping over
    the map. 
    
    this is a 'naive' function intended for testing against vectorized and 
    considerably faster pair-counting functions. 

    r_map and bins must have the same linear scale; typically this will be the
    centimorgan (cM). where it is useful to count site pairs whose left
    (lower-indexed) site is below some position, a left_bound may be applied;
    this is an index of r_map and counting ceases when it is reached.

    :param r_map: list or 1d array representing linear recombination map
    :param bins: list or 1d array of recombination-distance bin edges
    :param left_bound: optional, default None. imposes a maximum index on the
        left locus
    :return: 1d array of shape (len(bins) - 1) holding site pair counts
    """
    if not llim:
        llim = len(rmap)
    num_bins = len(bins) - 1
    num_pairs = np.zeros(len(bins) - 1, dtype=int)
    for i, rl in enumerate(rmap[:llim]):
        for rr in rmap[i + 1:]:
            distance = rr - rl
            bin_idx = bisect(bins, distance) - 1
            if bin_idx < num_bins:
                if bin_idx >= 0:
                    num_pairs[bin_idx] += 1
    num_pairs = ma.array(num_pairs, mask=num_pairs == 0)
    return num_pairs


def _get_num_pairs_arr(rmap, bins, llim=None):
    """
    """
    if not llim:
        llim = len(rmap)
    num_bins = len(bins) - 1
    nums_arr = np.zeros((llim, len(bins) - 1), dtype=int)
    for i, rl in enumerate(rmap[:llim]):
        for rr in rmap[i + 1:]:
            distance = rr - rl
            bin_idx = bisect(bins, distance) - 1
            if bin_idx < num_bins:
                if bin_idx >= 0:
                    nums_arr[i, bin_idx] += 1
    return nums_arr


def _count_sums_prods(umap, rmap, bins, llim=None):
    """
    
    """
    if len(umap) != len(rmap):
        raise ValueError('umap and rmap have mismatched lengths')
    if not llim:
        llim = len(rmap) 
    num_bins = len(bins) - 1
    sums_prods = np.zeros(len(bins) - 1, dtype=float)
    for i, (rl, ul) in enumerate(zip(rmap[:llim], umap[:llim])):
        for rr, ur in zip(rmap[i + 1:], umap[i + 1:]):
            distance = rr - rl
            bin_idx = bisect(bins, distance) - 1
            if bin_idx < num_bins:
                if bin_idx >= 0:
                    sums_prods[bin_idx] += ul * ur
    sums_prods = ma.array(sums_prods, mask=sums_prods == 0)
    return sums_prods


def count_num_pairs(rmap, bins, llim=None, verbosity=1e6):
    """
    get counts of site pairs binned by their recombination distance. 

    :param rmap: 1d array of recombination map values for each locus, in cM
    :param bins: 1d array of recombination bin edges, in cM
    :param llim: optional, default None. if provided, imposes a maximum index
        for the left locus
    """
    if not llim:
        llim = len(rmap)
    cumulative_nums = np.zeros(len(bins), dtype=int)
    for i, rl in enumerate(rmap[:llim]):
        edges = np.searchsorted(rmap[i + 1:], rl + bins)
        cumulative_nums += edges
        if i % verbosity == 0 and i > 0:
            print(util.get_time(), f'num pairs computed at site {i}')
    _num_pairs = np.diff(cumulative_nums)
    num_pairs = ma.array(_num_pairs, mask=_num_pairs == 0)
    return num_pairs


def compute_weight_facs(positions, rmap, umap, bins, windows, mean_u=None):
    """
    compute the denominator for the weighted H2 statistic, which is scaled by
    the average product of mutation rates in each bin.

    :param rmap: 1d array holding the recombination map in cM
    :param umap: 1d array of mutation map values
    :param bins: 1d array of recombination distance bin edges, in cM
    :param llim: optional, default None. maximum index for left loci
    :param mean_u: optional, default None. if provided, the square of this
        quantity is treated as the across-bins average mutation rate product
    """
    # compute the normalizing factor for the u-weighted H2 statistic
    num_windows = len(windows)
    num_bins = len(bins) - 1
    #num_pairs = np.zeros((num_windows, num_bins))
    u_prods = np.zeros((num_windows, num_bins))
    for w, (start, end, bound) in enumerate(windows):
        lrstart = np.searchsorted(positions, start)
        rlim = np.searchsorted(positions, bound)
        llim = np.searchsorted(positions[lrstart:], end)
        #num_pairs[w] = count_num_pairs(rmap[lrstart:rlim], bins, llim=llim)
        u_prods[w] = compute_uu_sums(
            rmap[lrstart:rlim], umap[lrstart:rlim], bins, llim=llim
        )
    #tot_prod = u_prods.sum()
    #tot_pairs = num_pairs.sum()
    # mean_prod = tot_prod / tot_pairs
    #facs = u_prods / mean_prod
    #print('tot_prod', tot_prod)
    #print('tot_pairs', tot_pairs)
    #facs = u_prods * tot_pairs / tot_prod



    tot_uu = get_chromosome_uu(umap)
    num_sites = len(umap)
    tot_num_pairs = num_sites * (num_sites - 1) // 2
    mean_uu = tot_uu / tot_num_pairs
    facs = u_prods / mean_uu

    print('tot uu', tot_uu)
    print('tot num sites', tot_num_pairs)
    print('mean uu', mean_uu)


    """
    tot_u_prods = bin_u_prods.sum()
    bin_u_prod_means = bin_u_prods / num_pairs
    if not mean_u:
        mean_u_prod = tot_u_prods / num_pairs.sum() 
    else:
        mean_u_prod = mean_u ** 2
    u_ratio = mean_u_prod / bin_u_prod_means
    facs = num_pairs * u_ratio
    """
    return facs


def compute_uu_sums(rmap, umap, bins, llim=None, verbosity=1e6):
    """
    """
    # compute sums of products of left * right locus mutation rates
    if len(umap) != len(rmap):
        raise ValueError('rmap length mismatches umap')
    if not llim:
        llim = len(rmap)
    cum_umap = np.cumsum(umap)
    cum_products = np.zeros(len(bins), dtype=float)
    for i, (rl, ul) in enumerate(zip(rmap[:llim], umap[:llim])):
        if ul > 0:
            edges = np.searchsorted(rmap[i + 1:], rl + bins)
            cum_products += ul * cum_umap[i:][edges]
            if i % verbosity == 0 and i > 0:
                print(util.get_time(), f'u prods computed at site {i}')
    _products = np.diff(cum_products)
    products = ma.array(_products, mask=_products == 0)
    return products
 

def get_chromosome_uu(umap):
    # get the total sum of u * u site products on a chromosome
    cum_umap = np.cumsum(umap)
    cum_uvals = cum_umap[-1] - cum_umap
    tot_uu = (cum_uvals * umap).sum()
    return tot_uu
