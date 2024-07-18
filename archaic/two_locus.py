"""

"""


import gzip
import numpy as np
from archaic import one_locus
from archaic import utils


"""
Read files
"""


def read_map_file(map_fname, map_col="Map(cM)"):

    if ".gz" in map_fname:
        open_fxn = gzip.open
    else:
        open_fxn = open
    file = open_fxn(map_fname, "rb")
    header = file.readline()
    file.close()
    header_fields = header.decode().strip('\n').split()
    if "Position(bp)" in header_fields:
        pos_idx = header_fields.index("Position(bp)")
    else:
        raise ValueError("There must be a 'Position(bp)' column!")
    if map_col in header_fields:
        map_idx = header_fields.index(map_col)
    else:
        raise ValueError(f"There must be a '{map_col}' column!")
    map_positions = []
    map_vals = []
    with open_fxn(map_fname, "rb") as file:
        for line_b in file:
            line = line_b.decode()
            fields = line.strip('\n').split('\t')
            if "Position(bp)" not in fields:
                map_positions.append(int(fields[pos_idx]))
                map_vals.append(float(fields[map_idx]))
    map_positions = np.array(map_positions)
    map_vals = np.array(map_vals)
    return map_positions, map_vals


def get_r_map(map_fname, positions, map_col="Map(cM)"):

    map_pos, map_vals = read_map_file(map_fname, map_col=map_col)
    if np.any(positions < map_pos[0]):
        print("There are positions below map start")
    if np.any(positions > map_pos[-1]):
        print("There are positions beyond map end")
    vals = np.interp(
        positions, map_pos, map_vals, left=map_vals[0], right=map_vals[-1]
    )
    return vals


"""
Map transformations
"""


def map_function(r_vals):
    # r > d
    return -50 * np.log(1 - 2 * r_vals)


"""
Compute H2 statistics
"""


def count_site_pairs(
    r_map,
    r_bins,
    positions=None,
    window=None,
    vectorized=True,
    bp_thresh=0,
    upper_bound=None,
    verbose=2
):
    # check whether there are actually any sites
    if len(r_map) == 0:
        return np.zeros(len(r_bins) - 1)
    # to use a distance threshold, we need positions
    if bp_thresh:
        if not np.any(positions):
            raise ValueError("You must provide positions to use bp_thresh!")
    d_bins = map_function(r_bins)
    if window is not None:
        if positions is None:
            raise ValueError("You must provide positions to use a window!")
        l_start, l_stop = np.searchsorted(positions, window)
        if upper_bound:
            r_stop = np.searchsorted(positions, upper_bound)
        else:
            max_d = r_map[l_stop - 1] + d_bins[-1]
            r_stop = np.searchsorted(r_map, max_d)
        r_map = r_map[l_start:r_stop]
        if bp_thresh:
            positions = positions[l_start:r_stop]
    else:
        l_start = 0
        l_stop = r_stop = len(r_map)
    n_left_loci = l_stop - l_start
    cum_counts = np.zeros(len(d_bins), dtype=np.int64)
    if vectorized:
        edges = r_map[:n_left_loci, np.newaxis] + d_bins[np.newaxis, :]
        counts = np.searchsorted(r_map, edges)
        cum_counts = counts.sum(0)
        n_site_pairs = np.diff(cum_counts)
        # correction on lowest bin
        if r_bins[0] == 0:
            n_redundant = np.sum(
                np.arange(n_left_loci)
                - np.searchsorted(r_map, r_map[:n_left_loci])
            ) + n_left_loci
            n_site_pairs[0] -= n_redundant
    else:
        for i in np.arange(n_left_loci):
            if bp_thresh > 0:
                j = np.searchsorted(positions, positions[i] + bp_thresh + 1)
            else:
                j = i + 1
            _bins = d_bins + r_map[i]
            cum_counts += np.searchsorted(r_map[j:], _bins)
            if i % 1e6 == 0:
                if n_left_loci > 1e6 and verbose == 2:
                    print(utils.get_time(), f"locus {i} of {n_left_loci} loci")
        n_site_pairs = np.diff(cum_counts)
    return n_site_pairs


def count_H2(
    genotypes,
    map_vals,
    r_bins,
    positions=None,
    window=None,
    vectorized=False,
    bp_thresh=0,
    upper_bound=None,
    verbose=1
):

    het_idx = one_locus.get_het_idx(genotypes)
    het_map_vals = map_vals[het_idx]
    if np.any(positions):
        het_positions = positions[het_idx]
    else:
        het_positions = None
    H2_counts = count_site_pairs(
        het_map_vals,
        r_bins,
        positions=het_positions,
        window=window,
        vectorized=vectorized,
        bp_thresh=bp_thresh,
        upper_bound=upper_bound,
        verbose=verbose
    )
    return H2_counts


def count_two_sample_H2(
    genotypes_x,
    genotypes_y,
    map_vals,
    r_bins,
    positions=None,
    window=None,
    bp_thresh=0,
    upper_bound=None
):
    # unphased, of course
    if bp_thresh:
        if not np.any(positions):
            raise ValueError("You must provide positions to use bp_thresh!")
    d_bins = map_function(r_bins)
    if np.any(window):
        if not np.any(positions):
            raise ValueError("You must provide positions to use a window!")
        l_start, l_stop = one_locus.get_window_bounds(window, positions)
        if upper_bound:
            r_stop = np.searchsorted(positions, upper_bound)
        else:
            max_d = map_vals[l_stop - 1] + d_bins[-1]
            r_stop = np.searchsorted(map_vals, max_d)
        map_vals = map_vals[l_start:r_stop]
        genotypes_y = genotypes_y[l_start:r_stop]
        genotypes_x = genotypes_x[l_start:r_stop]
        if bp_thresh:
            positions = positions[l_start:r_stop]
    else:
        l_start = 0
        l_stop = len(map_vals)
    n_left_loci = l_stop - l_start
    right_lims = np.searchsorted(map_vals, map_vals + d_bins[-1])
    site_Hxy = one_locus.get_site_Hxy(genotypes_x, genotypes_y)
    allowed_Hxy = np.array([0.25, 0.5, 0.75, 1])
    precomputed_H2xy = np.cumsum(allowed_Hxy[:, np.newaxis] * site_Hxy, axis=1)
    cum_counts = np.zeros(len(d_bins), dtype=np.float64)
    for i in np.arange(n_left_loci):
        if bp_thresh > 0:
            j_min = np.searchsorted(positions, positions[i] + bp_thresh + 1)
        else:
            j_min = i + 1
        j_max = right_lims[i]
        left_Hxy = site_Hxy[i]
        if left_Hxy > 0:
            _bins = d_bins + map_vals[i]
            edges = np.searchsorted(map_vals[j_min:j_max], _bins)
            select = np.searchsorted(allowed_Hxy, left_Hxy)
            locus_H2xy = precomputed_H2xy[select, i:j_max]
            cum_counts += locus_H2xy[edges]
    H2 = np.diff(cum_counts)
    return H2


def get_two_chromosome_H2(site_counts, H_counts):
    # all in one r-bin; 0.5. iterates over all pairs. returns H2, not a count
    # worth a rewrite
    n = len(site_counts)
    if len(H_counts) != n:
        raise ValueError("length mismatch")
    n_pairs = int(n * (n - 1) / 2)
    H2 = np.zeros(n_pairs)
    for i, (j, k) in enumerate(utils.get_pair_idxs(n)):
        site_pair_count = site_counts[j] * site_counts[k]
        H2_count = H_counts[j] * H_counts[k]
        H2[i] = H2_count / site_pair_count
    return H2


"""
Parsing files
"""


def get_one_sample_H2(
    genotypes,
    r_map,
    r_bins,
    positions,
    windows,
    bounds
):
    #
    site_H = genotypes[:, 0] != genotypes[:, 1]
    H_idx = np.nonzero(site_H)[0]
    H_r_map = r_map[H_idx]
    H_positions = positions[H_idx]
    H2 = np.zeros((len(windows), len(r_bins) - 1))
    for k, window in enumerate(windows):
        H2[k] = count_site_pairs(
            H_r_map,
            r_bins,
            positions=H_positions,
            window=window,
            upper_bound=bounds[k]
        )
    return H2


def get_two_sample_H2(
    genotypes_x,
    genotypes_y,
    r_map,
    r_bins,
    positions,
    windows,
    bounds
):
    #
    H2 = np.zeros((len(windows), len(r_bins) - 1))
    for k, window in enumerate(windows):
        H2[k] = count_two_sample_H2(
            genotypes_x,
            genotypes_y,
            r_map,
            r_bins,
            positions=positions,
            window=window,
            upper_bound=bounds[k]
        )
    return H2


def compute_H2(
    genotypes,
    genotype_positions,
    positions,
    r_map,
    r_bins,
    windows=None,
    bounds=None,
    sample_mask=None
):
    #
    if windows is None:
        windows = np.array([[positions[0], positions[-1] + 1]])
    if bounds is None:
        bounds = np.array([positions[-1] + 1])
    if sample_mask is not None:
        genotypes = genotypes[:, sample_mask]
    # site pairs
    n_site_pairs = np.zeros((len(windows), len(r_bins) - 1))
    for i, window in enumerate(windows):
        n_site_pairs[i] = count_site_pairs(
            r_map,
            r_bins,
            positions=positions,
            window=window,
            vectorized=True
        )
        print(
            utils.get_time(),
            f'n site pairs parsed for window {i}'
        )
    genotype_r_map = r_map[np.searchsorted(positions, genotype_positions)]
    n_samples = genotypes.shape[1]
    idxs = [(i, j) for i in range(n_samples) for j in np.arange(i, n_samples)]
    H2 = np.zeros((len(windows), len(idxs), len(r_bins) - 1))
    for i, (x, y) in enumerate(idxs):
        if x == y:
            H2[:, i] = get_one_sample_H2(
                genotypes[:, x],
                genotype_r_map,
                r_bins,
                genotype_positions,
                windows,
                bounds
            )
            print(
                utils.get_time(),
                f'H2 parsed for sample {x}'
            )
        else:
            H2[:, i] = get_two_sample_H2(
                genotypes[:, x],
                genotypes[:, y],
                genotype_r_map,
                r_bins,
                genotype_positions,
                windows,
                bounds
            )
            print(
                utils.get_time(),
                f'H2 parsed for samples {(x, y)}'
            )
    print(
        utils.get_time(),
        f'H2 parsed for {n_samples} samples '
        f'at {len(positions)} sites in {len(windows)} windows'
    )
    return n_site_pairs, H2





def parse_pair_counts(pos_map, pos, windows, bounds, r_bins, thresh=0):
    # count the number of site pairs in recombination bins
    b = len(r_bins) - 1
    site_pair_counts = np.zeros((len(windows), b), dtype=np.int64)
    for i, window in enumerate(windows):
        site_pair_counts[i] = count_site_pairs(
            pos_map,
            r_bins,
            positions=pos,
            window=window,
            upper_bound=bounds[i],
            bp_thresh=thresh,
            vectorized=True,
            verbose=1
        )
    return site_pair_counts


def parse_H2_counts(genotypes, vcf_map, pos, windows, bounds, r_bins, thresh=0):
    # count one and two sample H2
    # returns array of shape (n_samples + n_pairs, n_windows, n_bins)
    n_samples = genotypes.shape[1]
    n_pairs = utils.n_choose_2(n_samples)
    n = n_samples + n_pairs
    b = len(r_bins) - 1
    counts = np.zeros((n, len(windows), b), dtype=np.int64)
    for i in range(n_samples):
        for j, window in enumerate(windows):
            counts[i, j] = count_H2(
                genotypes[:, i],
                vcf_map,
                r_bins,
                positions=pos,
                window=window,
                upper_bound=bounds[j],
                bp_thresh=thresh,
                verbose=0
            )
    print(utils.get_time(), f"one sample H2 parsed")
    for i, (i_x, i_y) in enumerate(utils.get_pair_idxs(n_samples)):
        i += n_samples
        for j, window in enumerate(windows):
            counts[i, j] = count_H2xy(
                genotypes[:, i_x],
                genotypes[:, i_y],
                vcf_map,
                r_bins,
                positions=pos,
                upper_bound=bounds[j],
                window=window,
                verbose=False
            )
    print(utils.get_time(), f"two sample H2 parsed")
    return counts
