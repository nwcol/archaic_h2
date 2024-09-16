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





