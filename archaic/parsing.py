"""
Functions for parsing statistics from .vcf files
"""
import numpy as np
import time

from archaic import masks, utils


_default_r_bins = np.logspace(-6, -2, 17)


"""
computing one-locus statistics
"""


def get_one_sample_H(genotypes, vcf_positions, windows):
    # compute H sums for a single diploid sample
    site_H = genotypes[:, 0] != genotypes[:, 1]
    num_H = np.zeros(len(windows))
    for i, window in enumerate(windows):
        lower, upper = np.searchsorted(vcf_positions, window)
        num_H[i] = site_H[lower:upper].sum()
    return num_H


def get_two_sample_H(genotypes_x, genotypes_y, vcf_positions, windows):
    # compute H sums for two diploid samples
    site_H = get_two_sample_site_H(genotypes_x, genotypes_y)
    num_H = np.zeros(len(windows))
    for i, window in enumerate(windows):
        lower, upper = np.searchsorted(vcf_positions, window)
        num_H[i] = site_H[lower:upper].sum()
    return num_H


def get_multi_sample_H(genotype_arr, vcf_positions, windows):


    return 0


def get_two_sample_site_H(genotypes_x, genotypes_y):
    # compute probabilities of sampling a distinct allele from x and y at each
    site_H = (
        np.sum(genotypes_x[:, 0][:, np.newaxis] != genotypes_y, axis=1)
        + np.sum(genotypes_x[:, 1][:, np.newaxis] != genotypes_y, axis=1)
    ) / 4
    return site_H


def compute_H(
    positions,
    genotypes,
    vcf_positions,
    windows=None,
    sample_mask=None,
    verbose=True
):
    # compute H across a chromosome in one or more windows
    if windows is None:
        windows = np.array([[positions[0], positions[-1] + 1]])
    if sample_mask is not None:
        genotypes = genotypes[:, sample_mask]
    n_sites = np.diff(np.searchsorted(positions, windows))[:, 0]
    _, n_samples, __ = genotypes.shape
    H = np.zeros((len(windows), utils.n_choose_2(n_samples) + n_samples))
    k = 0

    for i in range(n_samples):
        for j in range(i, n_samples):
            if i == j:
                H[:, k] = get_one_sample_H(
                    genotypes[:, i], vcf_positions, windows
                )
            else:
                H[:, k] = get_two_sample_H(
                    genotypes[:, i], genotypes[:, j], vcf_positions, windows
                )
            k += 1

    if verbose:
        print(
            utils.get_time(),
            f'computed H for {n_samples} samples '
            f'at {n_sites.sum()} sites in {len(windows)} windows'
        )
    return n_sites, H


def compute_SFS(variant_file, ref_as_ancestral=False):
    # variants needs the ancestral allele field in info
    # ref_is_ancestral=True is for simulated data which lacks INFO=AA
    genotypes = variant_file.genotypes
    refs = variant_file.refs
    alts = variant_file.alts
    if ref_as_ancestral:
        ancs = refs
    else:
        ancs = variant_file.ancestral_alleles
    sample_ids = variant_file.sample_ids
    n = len(sample_ids)
    SFS = np.zeros([3] * n, dtype=np.int64)
    n_triallelic = 0
    n_mismatch = 0
    for i in range(len(variant_file)):
        ref = refs[i]
        alt = alts[i]
        anc = ancs[i]
        segregating = [ref] + alt.split(',')
        if len(segregating) > 2:
            n_triallelic += 1
            # we ignore multiallelic sites
            continue
        if anc not in segregating:
            n_mismatch += 1
            # ancestral allele isn't represented in the sample
            continue
        if ref == anc:
            SFS_idx = tuple(genotypes[i].sum(1))
        elif alt == anc:
            SFS_idx = tuple(2 - genotypes[i].sum(1))
        else:
            print('...')
            SFS_idx = None
        SFS[SFS_idx] += 1
    print(
        utils.get_time(),
        f'{n_triallelic} multiallelic sites, '
        f'{n_mismatch} sites lacking ancestral allele'
    )
    return SFS, sample_ids


"""
computing two-locus statistics
"""


def count_site_pairs(
    r_map,
    bins,
    left_bound=None,
    thresh=0,
    positions=None
):
    #
    if len(r_map) < 2:
        return np.zeros(len(bins) - 1)

    if not left_bound:
        left_bound = len(r_map)

    cM_bins = utils.map_function(bins)
    site_edges = r_map[:left_bound, np.newaxis] + cM_bins[np.newaxis, :]
    counts = np.searchsorted(r_map, site_edges)
    cum_counts = counts.sum(0)
    num_pairs = np.diff(cum_counts)

    if bins[0] == 0:
        # correction for 0-distance 'pre-counting'
        redundant_count = np.sum(
            np.arange(left_bound) - np.searchsorted(r_map, r_map[:left_bound])
        )
        # correction for self-counting
        redundant_count += left_bound
        num_pairs[0] -= redundant_count
    return num_pairs


def get_site_pairs(
    positions,
    windows,
    bounds,
    r_map,
    bins
):
    #
    num_sites = np.zeros((len(windows), len(bins) - 1))
    for i, (window, bound) in enumerate(zip(windows, bounds)):
        start = np.searchsorted(positions, window[0])
        right_bound = np.searchsorted(positions, bound)
        _r_map = r_map[start:right_bound]
        left_bound = np.searchsorted(positions[start:], window[1])
        num_sites[i] = count_site_pairs(_r_map, bins, left_bound=left_bound)
    return num_sites


def get_one_sample_H2(
    genotypes,
    positions,
    windows,
    bounds,
    r_map,
    bins,
):
    site_H = genotypes[:, 0] != genotypes[:, 1]
    H_idx = np.nonzero(site_H)[0]
    _r_map = r_map[H_idx]
    _positions = positions[H_idx]
    num_H2 = np.zeros((len(windows), len(bins) - 1))
    for i, (window, bound) in enumerate(zip(windows, bounds)):
        start = np.searchsorted(_positions, window[0])
        right_bound = np.searchsorted(_positions, bound)
        __r_map = _r_map[start:right_bound]
        left_bound = np.searchsorted(_positions[start:], window[1])
        num_H2[i] = count_site_pairs(__r_map, bins, left_bound=left_bound)
    return num_H2


def count_two_sample_H2(
    genotypes,
    r_map,
    bins,
    left_bound=None,
    thresh=0,
    positions=None
):
    # unphased
    # make sure that we don't have an empty map array
    if len(r_map) < 2:
        return np.zeros(len(bins) - 1)

    if not left_bound:
        left_bound = len(r_map)

    cM_bins = utils.map_function(bins)
    right_lims = np.searchsorted(r_map, r_map + cM_bins[-1])
    site_H = get_two_sample_site_H(genotypes[:, 0], genotypes[:, 1])

    # precompute site two-sample H
    # the values 0.25, 0.75 occur only when we have triallelic sites
    allowed = np.array([0.25, 0.5, 0.75, 1])
    precomputed_H = np.cumsum(allowed[:, np.newaxis] * site_H, axis=1)
    cum_counts = np.zeros(len(cM_bins), dtype=np.float64)

    for i in np.arange(left_bound):
        if thresh > 0:
            j_min = np.searchsorted(positions, positions[i] + thresh + 1)
        else:
            j_min = i + 1

        j_max = right_lims[i]
        left_H = site_H[i]

        if left_H > 0:
            _bins = cM_bins + r_map[i]
            edges = np.searchsorted(r_map[j_min:j_max], _bins)
            select = np.searchsorted(allowed, left_H)
            locus_H2 = precomputed_H[select, i:j_max]
            cum_counts += locus_H2[edges]

    num_H2 = np.diff(cum_counts)
    return num_H2


def get_two_sample_H2(
    genotypes,
    positions,
    windows,
    bounds,
    r_map,
    bins,
):
    #
    num_H2 = np.zeros((len(windows), len(bins) - 1))
    for i, (window, bound) in enumerate(zip(windows, bounds)):
        start = np.searchsorted(positions, window[0])
        right_bound = np.searchsorted(positions, bound)
        _r_map = r_map[start:right_bound]
        _genotypes = genotypes[start:right_bound]
        left_bound = np.searchsorted(positions[start:], window[1])
        num_H2[i] = count_two_sample_H2(
            _genotypes, _r_map, bins, left_bound=left_bound
        )
    return num_H2


def get_cross_chromosome_H2(site_counts, H_counts):
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


def compute_H2(
    positions,
    genotypes,
    vcf_positions,
    r_map,
    bins=None,
    windows=None,
    bounds=None,
    sample_mask=None,
    verbose=True
):
    # across a chromosome
    # num pairs has shape (n_windows, n_bins)
    # num H2 has shape (n_windows, n_samples + n_pairs, n_bins)
    if windows is None:
        windows = np.array([[positions[0], positions[-1] + 1]])
    if bounds is None:
        bounds = np.array([positions[-1] + 1])
    if sample_mask is not None:
        genotypes = genotypes[:, sample_mask]
    if bins is None:
        bins = _default_r_bins

    num_pairs = get_site_pairs(positions, windows, bounds, r_map, bins)

    vcf_r_map = r_map[np.searchsorted(positions, vcf_positions)]
    length, n_samples, _ = genotypes.shape
    num_H2 = np.zeros(
        (len(windows), utils.n_choose_2(n_samples) + n_samples, len(bins) - 1)
    )
    k = 0

    for i in range(n_samples):
        for j in range(i, n_samples):
            if i == j:
                num_H2[:, k] = get_one_sample_H2(
                    genotypes[:, i],
                    vcf_positions,
                    windows,
                    bounds,
                    vcf_r_map,
                    bins
                )
            else:
                num_H2[:, k] = get_two_sample_H2(
                    genotypes[:, [i, j]],
                    vcf_positions,
                    windows,
                    bounds,
                    vcf_r_map,
                    bins,
                )
            k += 1

    print(
        utils.get_time(),
        f'H2 parsed for {n_samples} samples '
        f'at {len(positions)} sites in {len(windows)} windows'
    )
    return num_pairs, num_H2


def count_scaled_site_pairs(
    r_map,
    u_map,
    bins,
    max_idx=None,
    positions=None,
    thresh=None,
    verbosity=1e6,
    discretization=30
):
    #
    if not max_idx:
        max_idx = len(r_map)

    cM_bins = utils.map_function(bins)

    # optimize by capping right index
    right_limits = np.searchsorted(r_map, r_map + cM_bins[-1])
    right_limits[right_limits > max_idx] = max_idx
    cum_scaled_num_pairs = np.zeros(len(bins))

    _bins = r_map[:max_idx, np.newaxis] + cM_bins[np.newaxis, :]
    edges = np.searchsorted(r_map, _bins)
    if bins[0] == 0:
        idx = edges[:, 0] <= np.arange(len(edges))
        edges[idx, 0] = np.arange(len(edges))[idx] + 1

    discrete_u = np.logspace(
        np.log10(u_map.min()), np.log10(u_map.max() * 1.001), discretization
    )
    precomputed = dict()
    for i, u in enumerate(discrete_u):
        vals = np.cumsum(discrete_u[i] * u_map)
        vals = np.append(vals, vals[-1])
        precomputed[i] = vals

    for i in np.arange(max_idx):
        #scales = u_map[i] * u_map[i:max_idx]
        #cum_scales = np.cumsum(scales)
        #cum_scaled_num_pairs += cum_scales[edges[i]]

        idx = np.searchsorted(discrete_u, u_map[i])
        cum_scaled_num_pairs += precomputed[idx][edges[i]]

        if i % verbosity == 0:
            if i > 0:
                print(utils.get_time(), f'site pairs counted at site {i}')

    scaled_num_pairs = np.diff(cum_scaled_num_pairs)

    return scaled_num_pairs


def parse_scaled_H2(
    mask_fname,
    vcf_fname,
    map_fname,
    scale_fname,
    r_bins,
    windows=None,
    bounds=None,
    scale_name='rates',
):
    #
    mask = masks.Mask.from_bed_file(mask_fname)
    mask_positions = mask.positions
    variant_file = one_locus.VariantFile(vcf_fname, mask=mask)
    genotype_arr = variant_file.genotypes
    vcf_positions = variant_file.positions
    r_map = two_locus.get_r_map(map_fname, mask_positions)
    vcf_map = two_locus.get_r_map(map_fname, vcf_positions)
    print(utils.get_time(), 'files loaded')

    scale_file = np.load(scale_fname)
    scale_positions = scale_file['positions']
    _scale = 1 / scale_file['rates']
    scale = _scale[np.searchsorted(scale_positions, mask_positions)]
    vcf_scale = _scale[np.searchsorted(scale_positions, vcf_positions)]
    print(utils.get_time(), 'rate files loaded')

    length, n_samples, _ = genotype_arr.shape
    n_windows = len(windows)

    # one-locus H
    idxs = [(i, j) for i in range(n_samples) for j in np.arange(i, n_samples)]
    n_sites = np.zeros(n_windows)
    H_counts = np.zeros((n_windows, len(idxs)))

    for k, window in enumerate(windows):
        start, end = np.searchsorted(mask_positions, window)
        _start, _end = np.searchsorted(vcf_positions, window)
        n_sites[k] = scale[start:end].sum()

        for i, (x, y) in enumerate(idxs):
            if x != y:
                continue
            sequences = genotype_arr[_start:_end, x]
            indicator = sequences[:, 0] != sequences[:, 1]
            H_counts[k, i] = (indicator * vcf_scale[_start:_end]).sum()
    print(utils.get_time(), 'one-locus H computed')

    # two-locus H
    n_site_pairs = np.zeros((n_windows, len(r_bins) - 1))
    H2_counts = np.zeros((n_windows, len(idxs), len(r_bins) - 1))

    for k, window in enumerate(windows):
        start, end = np.searchsorted(mask_positions, window)
        max_idx = np.searchsorted(mask_positions, bounds[k])

        _start, _end = np.searchsorted(vcf_positions, window)
        _max_idx = np.searchsorted(vcf_positions, bounds[k])

        n_site_pairs[k] = count_u_scaled_site_pairs(
            r_map[start:],
            scale[start:],
            r_bins,
            max_idx=max_idx
        )
        for i, (x, y) in enumerate(idxs):
            if x != y:
                continue
            sequences = genotype_arr[_start:_end, x]
            indicator = sequences[_start:, 0] != sequences[_start:, 1]
            H2_counts[0, i] = count_u_scaled_site_pairs(
                vcf_map[_start:][indicator],
                vcf_scale[_start:][indicator],
                r_bins,
                max_idx=_max_idx
            )
    print(utils.get_time(), 'two-locus H computed')

    ids = variant_file.sample_ids
    n = len(ids)
    ids = [(ids[i], ids[j]) for i in np.arange(n) for j in np.arange(i, n)]
    ret = dict(
        n_sites=n_sites[np.newaxis],
        H_counts=H_counts,
        n_site_pairs=n_site_pairs[np.newaxis],
        H2_counts=H2_counts,
        r_bins=r_bins,
        ids=ids
    )
    return ret


def parse_H(
    mask_fname,
    vcf_fname,
    windows
):
    # from file
    mask_regions = utils.read_mask_file(mask_fname)
    mask_positions = utils.mask_to_positions(mask_regions)
    sample_ids, vcf_positions, genotype_arr = \
        utils.read_vcf_genotypes(vcf_fname, mask_regions)
    num_sites, num_H = compute_H(
        mask_positions,
        genotype_arr,
        vcf_positions,
        windows=windows
    )
    return num_sites, num_H


def parse_H2(
    mask_fname,
    vcf_fname,
    map_fname,
    windows=None,
    bounds=None,
    bins=None
):
    #
    t0 = time.time()
    mask_regions = utils.read_mask_file(mask_fname)
    mask_positions = utils.mask_to_positions(mask_regions)
    sample_ids, vcf_positions, genotype_arr = \
        utils.read_vcf_genotypes(vcf_fname, mask_regions)
    r_map = utils.read_map_file(map_fname, mask_positions)
    print(utils.get_time(), "files loaded")

    num_sites, num_H = compute_H(
        mask_positions,
        genotype_arr,
        vcf_positions,
        windows=windows
    )
    num_pairs, num_H2 = compute_H2(
        mask_positions,
        genotype_arr,
        vcf_positions,
        r_map,
        bins=bins,
        windows=windows,
        bounds=bounds
    )

    n = len(sample_ids)
    stat_ids = [
        (sample_ids[i], sample_ids[j]) for i in np.arange(n) for j in np.arange(i, n)
    ]
    stats = dict(
        ids=stat_ids,
        r_bins=bins,
        windows=windows,
        bounds=bounds,
        n_sites=num_sites,
        H_counts=num_H,
        n_site_pairs=num_pairs,
        H2_counts=num_H2
    )

    t = np.round(time.time() - t0, 0)
    chrom_num = utils.read_vcf_contig(vcf_fname)
    print(
        utils.get_time(),
        f'{len(mask_positions)} sites on '
        f'chromosome {chrom_num} parsed in\t{t} s')
    return stats


def bootstrap_H2(dics, n_iters=1000, bin_slice=None):
    # carry out bootstraps to get H, H2 distributions.
    # takes dictionaries as args
    n_sites = np.concatenate([dic['n_sites'] for dic in dics])
    H_counts = np.concatenate([dic['H_counts'] for dic in dics])
    n_site_pairs = np.concatenate([dic['n_site_pairs'] for dic in dics])
    H2_counts = np.concatenate([dic['H2_counts'] for dic in dics])

    if bin_slice is None:
        n_windows, n, n_bins = H2_counts.shape
        r_bins = dics[0]["r_bins"]
    else:
        start, stop = bin_slice
        n_windows, n, _ = H2_counts.shape
        r_bins = dics[0]['r_bins'][start:stop + 2]
        print(f'r_bins sliced to {r_bins}')
        n_bins = len(r_bins) - 1
        n_site_pairs = n_site_pairs[:, start:stop + 1]
        H2_counts = H2_counts[:, :, start:stop + 1]

    H_dist = np.zeros((n_iters, n))
    H2_dist = np.zeros((n_iters, n, n_bins))
    for i in range(n_iters):
        sample = np.random.randint(n_windows, size=n_windows)
        H_dist[i] = H_counts[sample].sum(0) / n_sites[sample].sum()
        H2_dist[i] = H2_counts[sample].sum(0) / n_site_pairs[sample].sum(0)

    H2_cov = np.zeros((n_bins, n, n))
    for i in range(n_bins):
        H2_cov[i, :, :] = np.cov(H2_dist[:, :, i], rowvar=False)

    ids = dics[0]["ids"]
    # we transpose some arrays for more desirable behavior in inference
    stats = dict(
        ids=ids,
        r_bins=r_bins,
        H_dist=H_dist,
        H_mean=H_dist.mean(0),
        H_cov=np.cov(H_dist, rowvar=False),
        H2_dist=H2_dist,
        H2_mean=H2_dist.mean(0).T,
        H2_cov=H2_cov
    )
    return stats


"""
SFS
"""


def parse_SFS(mask_fnames, vcf_fnames, out_fname, ref_as_ancestral=False):
    # takes many input files. assumes .vcfs are already masked!
    n_sites = 0
    for mask_fname in mask_fnames:
        mask = masks.read_mask_regions(mask_fname)
        _n_sites = masks.get_n_sites(mask)
        n_sites += _n_sites
        print(
            utils.get_time(),
            f'{_n_sites} positions parsed from {mask_fname}'
        )
    samples = one_locus.read_vcf_sample_names(vcf_fnames[0])
    n = len(samples)
    SFS = np.zeros([3] * n, dtype=np.int64)
    for vcf_fname in vcf_fnames:
        variants = one_locus.VariantFile(vcf_fname)
        _SFS, _samples = one_locus.parse_SFS(
            variants, ref_as_ancestral=ref_as_ancestral
        )
        if not np.all(_samples == samples):
            raise ValueError(f'sample mismatch: {samples}, {_samples}')
        SFS += _SFS
        print(
            utils.get_time(),
            f'{_SFS.sum()} variants parsed to SFS '
            f'on chrom {variants.chrom_num}'
        )
    arrs = dict(
        samples=samples,
        n_sites=n_sites,
        SFS=SFS
    )
    np.savez(out_fname, **arrs)


def parse_window_SFS():
    # implement

    return 0
