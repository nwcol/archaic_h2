"""
Functions for parsing statistics from .vcf files
"""
import numpy as np
import time

from archaic import masks, utils


_default_bins = np.logspace(-6, -2, 17)


"""
computing one-locus statistics
"""


def count_two_sample_site_H(gt_x, gt_y):
    # compute probabilities of sampling a distinct allele from x and y at each
    # locus
    _gt_x = gt_x[:, :, np.newaxis]
    _gt_y = gt_y[:, np.newaxis]
    site_H = (_gt_x != _gt_y).sum((2, 1)) / 4
    return site_H


def compute_H(
    positions,
    genotype_arr,
    vcf_positions,
    windows=None,
    sample_mask=None,
    verbose=True
):
    # compute H across a chromosome in one or more windows
    if windows is None:
        windows = np.array([[positions[0], positions[-1] + 1]])
    if sample_mask is not None:
        genotype_arr = genotype_arr[:, sample_mask]

    n_sites = np.diff(np.searchsorted(positions, windows))[:, 0]
    _, n_samples, __ = genotype_arr.shape
    num_H = np.zeros((len(windows), utils.n_choose_2(n_samples) + n_samples))
    k = 0

    for z, window in enumerate(windows):
        _start, _end = np.searchsorted(vcf_positions, window)
        for i in range(n_samples):
            for j in range(i, n_samples):
                if i == j:
                    gt = genotype_arr[_start:_end, i]
                    site_H = gt[:, 0] != gt[:, 1]
                    num_H[:, k] = site_H.sum()
                else:
                    gt_i = genotype_arr[_start:_end, i]
                    gt_j = genotype_arr[_start:_end, j]
                    site_H = count_two_sample_site_H(gt_i, gt_j)
                    num_H[:, k] = site_H.sum()
                k += 1

    return n_sites, num_H


def compute_scaled_H(
    positions,
    genotype_arr,
    vcf_positions,
    scale,
    windows=None
):
    # scale local H by the average mutation rate. scale is proportional to u
    if windows is None:
        windows = np.array([[positions[0], positions[-1] + 1]])

    vcf_scale = scale[np.searchsorted(positions, vcf_positions)]
    num_sites = np.zeros(len(windows))
    _, n_samples, __ = genotype_arr.shape
    num_H = np.zeros((len(windows), utils.n_choose_2(n_samples) + n_samples))

    for z, window in enumerate(windows):
        start, end = np.searchsorted(positions, window)
        # scale factor
        u_bar = scale[start:end].mean()
        n = end - start
        num_sites[z] = n / u_bar
        _start, _end = np.searchsorted(vcf_positions, window)
        k = 0

        for i in range(n_samples):
            for j in range(i, n_samples):
                if i == j:
                    gt = genotype_arr[_start:_end, i]
                    indicator = gt[:, 0] != gt[:, 1]
                    num_H[z, k] = (1 / vcf_scale[_start:_end][indicator]).sum()
                else:
                    pass
                k += 1

    return num_sites, num_H


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
        bins = _default_bins

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
    bins,
    scale,
    left_bound=None,
    verbosity=1e6
):
    #
    if not left_bound:
        left_bound = len(r_map)

    cM_bins = utils.map_function(bins)
    _bins = r_map[:left_bound, np.newaxis] + cM_bins[np.newaxis, :]
    edges = np.searchsorted(r_map, _bins)
    edges -= np.arange(left_bound)[:, np.newaxis]

    # a cheat- I don't fully understand this indexing issue
    edges[edges == len(edges)] -= 1

    # precompute scale products
    cum_scale = np.cumsum(scale)
    cum_scaled_num_pairs = np.zeros(len(bins))

    for i in np.arange(left_bound):
        edge_scales = cum_scale[edges[i]]
        cum_scaled_num_pairs += scale[i] * edge_scales
        if i % verbosity == 0:
            if i > 0:
                print(utils.get_time(), f'site pairs counted at site {i}')

    scaled_num_pairs = np.diff(cum_scaled_num_pairs)
    return scaled_num_pairs


"""
parsing statistics from file
"""


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
    t0 = time.time()
    mask_regions = utils.read_mask_file(mask_fname)
    mask_positions = utils.mask_to_positions(mask_regions)
    sample_ids, vcf_positions, genotype_arr = \
        utils.read_vcf_genotypes(vcf_fname, mask_regions)
    r_map = utils.read_map_file(map_fname, mask_positions)
    vcf_r_map = utils.read_map_file(map_fname, vcf_positions)
    print(utils.get_time(), "files loaded")

    scale_file = np.load(scale_fname)
    scale_positions = scale_file['positions']
    _scale = scale_file['rates']
    scale = _scale[np.searchsorted(scale_positions, mask_positions)]
    vcf_scale = _scale[np.searchsorted(scale_positions, vcf_positions)]
    print(utils.get_time(), 'scale files loaded')

    length, n_samples, _ = genotype_arr.shape
    n_windows = len(windows)

    # one-locus H
    n_sites = np.zeros(n_windows)
    H_counts = np.zeros((n_windows, utils.n_choose_2(n_samples) + n_samples))

    for z, window in enumerate(windows):
        start, end = np.searchsorted(mask_positions, window)
        _start, _end = np.searchsorted(vcf_positions, window)
        n_sites[z] = scale[start:end].sum()
        k = 0
        for i in range(n_samples):
            for j in range(i, n_samples):
                if i != j:
                    k += 1
                    continue
                sequences = genotype_arr[_start:_end, i]
                indicator = sequences[:, 0] != sequences[:, 1]
                H_counts[z, k] = (indicator * vcf_scale[_start:_end]).sum()
                k += 1

    print(utils.get_time(), 'one-locus H computed')

    # two-locus H
    n_site_pairs = np.zeros((n_windows, len(r_bins) - 1))
    H2_counts = np.zeros(
        (n_windows, utils.n_choose_2(n_samples) + n_samples, len(r_bins) - 1)
    )

    for z, (window, bound) in enumerate(zip(windows, bounds)):
        start = np.searchsorted(mask_positions, window[0])
        left_bound = np.searchsorted(mask_positions[start:], window[1])
        right_bound = np.searchsorted(mask_positions, bounds[z])
        n_site_pairs[z] = count_scaled_site_pairs(
            r_map[start:right_bound],
            r_bins,
            scale[start:right_bound],
            left_bound=left_bound
        )
        _start = np.searchsorted(vcf_positions, window[0])
        _right_bound = np.searchsorted(vcf_positions, bound)
        k = 0
        for i in range(n_samples):
            for j in range(i, n_samples):
                if i != j:
                    k += 1
                    continue
                sequences = genotype_arr[_start:_right_bound, i]
                indicator = sequences[:, 0] != sequences[:, 1]
                _left_bound = np.searchsorted(
                    vcf_positions[_start:_right_bound][indicator], window[1]
                )
                H2_counts[z, k] = count_scaled_site_pairs(
                    vcf_r_map[_start:_right_bound][indicator],
                    r_bins,
                    vcf_scale[_start:_right_bound][indicator],
                    left_bound=_left_bound
                )
                k += 1

    print(utils.get_time(), 'two-locus H computed')

    n = len(sample_ids)
    stat_ids = [
        (sample_ids[i], sample_ids[j]) for i in np.arange(n) for j in np.arange(i, n)
    ]
    ret = dict(
        n_sites=n_sites,
        H_counts=H_counts,
        n_site_pairs=n_site_pairs,
        H2_counts=H2_counts,
        r_bins=r_bins,
        ids=stat_ids
    )
    return ret


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


def sum_H2(*args, fancy=True):
    # sum a number of dictionaries together
    _ids = 'ids'
    _r_bins = 'r_bins'
    _n_sites = 'n_sites'
    _n_site_pairs = 'n_site_pairs'
    _n_H = 'H_counts'
    _n_H2 = 'H2_counts'
    stats = [_n_sites, _n_site_pairs, _n_H, _n_H2]

    template = args[0]

    ids = template[_ids]
    r_bins = template[_r_bins]
    for dic in args:
        if np.any(ids != dic[_ids]):
            raise ValueError(f'id mismatch in {dic}')
        if np.any(r_bins != dic[_r_bins]):
            raise ValueError(f'r bin mismatch in {dic}')

    ret = {_n_sites: [], _n_site_pairs: [], _n_H: [], _n_H2: []}

    for dic in args:
        for stat in stats:
            ret[stat].append(dic[stat])


    for stat in [_n_site_pairs, _n_H, _n_H2]:
        ret[stat] = np.vstack(ret[stat])
    for stat in [_n_sites]:
        ret[stat] = np.concatenate(ret[stat])

    ret[_ids] = ids
    ret[_r_bins] = r_bins

    if fancy:
        ret['window_H'] = ret[_n_H] / ret[_n_sites][:, np.newaxis]
        ret['window_H2'] = ret[_n_H2] / ret[_n_site_pairs][:, np.newaxis, :]
        ret['H'] = ret[_n_H].sum(0) / ret[_n_sites].sum(0)
        ret['H2'] = ret[_n_H2].sum(0) / ret[_n_site_pairs].sum(0)
        ret['std_H'] = np.std(ret['window_H'], axis=0)
        ret['std_H2'] = np.std(ret['window_H2'], axis=0)

    return ret


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


"""
computing H2 across chromosome arms or across chromosomes
"""


def compute_cross_arm_H2(chrom_dict, cen_idx):
    # returns a single 'r-bin'. cen_idx indexes first window after the
    # centromere
    H = 'H_counts'
    sites = 'n_sites'
    H2 = 'H2_counts'
    pairs = 'n_site_pairs'

    num_H2 = chrom_dict[H][:cen_idx].sum(0) * chrom_dict[H][cen_idx:].sum(0)
    num_pairs = \
        chrom_dict[sites][:cen_idx].sum() * chrom_dict[sites][cen_idx:].sum()

    expected_num_pairs = utils.n_choose_2(chrom_dict[sites].sum())
    assert num_pairs + chrom_dict[pairs].sum() == expected_num_pairs

    return num_H2, num_pairs


def compute_cross_chrom_H2(chrom_dicts):

    H = 'H_counts'
    sites = 'n_sites'

    n_chroms = len(chrom_dicts)
    n_pairs = utils.n_choose_2(n_chroms)
    _, n_stats = chrom_dicts[0]['H_counts'].shape
    num_H2 = np.zeros((n_pairs, n_stats))
    num_pairs = np.zeros(n_pairs)
    k = 0

    for i in range(n_chroms):
        for j in range(i + 1, n_chroms):
            chr_i = chrom_dicts[i]
            chr_j = chrom_dicts[j]
            num_H2[k] = chr_i[H].sum(0) * chr_j[H].sum(0)
            num_pairs[k] = chr_i[sites].sum() * chr_j[sites].sum()
            k += 1

    return num_pairs, num_H2
