"""
Functions for parsing statistics from .vcf files
"""
import numpy as np
import time

from archaic import utils


_default_bins = np.logspace(-6, -2, 17)


"""
computing one-locus statistics
"""


def get_two_sample_site_H(gts_x, gts_y):
    # compute probabilities of sampling a distinct allele from x and y at each
    # locus
    gts_x = gts_x[:, :, np.newaxis]
    gts_y = gts_y[:, np.newaxis]
    site_H = (gts_x != gts_y).sum((2, 1)) / 4
    return site_H


def compute_H(
    positions,
    genotype_arr,
    vcf_positions,
    windows=None,
    sample_mask=None,
    get_two_sample=True
):
    # compute H across a chromosome in one or more windows
    if windows is None:
        windows = np.array([[positions[0], positions[-1] + 1]])
    if sample_mask is not None:
        genotype_arr = genotype_arr[:, sample_mask]

    n_sites = np.diff(np.searchsorted(positions, windows))[:, 0]
    _, n_samples, __ = genotype_arr.shape

    if get_two_sample:
        n_stats = n_samples + utils.n_choose_2(n_samples)
    else:
        n_stats = n_samples

    num_H = np.zeros((len(windows), n_stats))

    for z, window in enumerate(windows):
        vcf_start, vcf_end = np.searchsorted(vcf_positions, window)
        k = 0

        for i in range(n_samples):
            for j in range(i, n_samples):
                if i == j:
                    gts = genotype_arr[vcf_start:vcf_end, i]
                    site_H = gts[:, 0] != gts[:, 1]
                    num_H[z, k] = site_H.sum()
                    k += 1
                else:
                    if not get_two_sample:
                        continue
                    gts_i = genotype_arr[vcf_start:vcf_end, i]
                    gts_j = genotype_arr[vcf_start:vcf_end, j]
                    site_H = get_two_sample_site_H(gts_i, gts_j)
                    num_H[z, k] = site_H.sum()
                    k += 1

    return n_sites, num_H


def compute_weighted_H(
    positions,
    genotype_arr,
    vcf_positions,
    weights,
    windows=None
):
    # scale local H by the average mutation rate. scale is proportional to u
    if windows is None:
        windows = np.array([[positions[0], positions[-1] + 1]])

    vcf_weights = weights[np.searchsorted(positions, vcf_positions)]
    num_sites = np.zeros(len(windows))
    _, n_samples, __ = genotype_arr.shape
    num_H = np.zeros((len(windows), utils.n_choose_2(n_samples) + n_samples))

    for w, window in enumerate(windows):
        start, end = np.searchsorted(positions, window)
        num_sites[w] = weights[start:end].sum()

        vcf_start, vcf_end = np.searchsorted(vcf_positions, window)
        win_vcf_weights = vcf_weights[vcf_start:vcf_end]
        k = 0

        for i in range(n_samples):
            for j in range(i, n_samples):
                if i == j:
                    gts = genotype_arr[vcf_start:vcf_end, i]
                    indicator = gts[:, 0] != gts[:, 1]
                    num_H[w, k] = win_vcf_weights[indicator].sum()
                else:
                    gts_i = genotype_arr[vcf_start:vcf_end, i]
                    gts_j = genotype_arr[vcf_start:vcf_end, j]
                    site_H = get_two_sample_site_H(gts_i, gts_j)
                    num_H[w, k] = (win_vcf_weights * site_H).sum()
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
    left_bound=None
):
    #
    if len(r_map) < 2:
        return np.zeros(len(bins) - 1)

    if not left_bound:
        left_bound = len(r_map)
    else:
        if left_bound > len(r_map):
            print('f')

    site_bins = r_map[:left_bound, np.newaxis] + bins[np.newaxis, :]
    bin_edges = np.searchsorted(r_map, site_bins)
    too_low = bin_edges[:, 0] <= np.arange(left_bound)
    bin_edges[too_low, 0] = np.arange(left_bound)[too_low] + 1
    num_pairs = np.diff(bin_edges.sum(0))
    return num_pairs


def count_weighted_site_pairs(
    weights,
    r_map,
    bins,
    left_bound=None,
    verbosity=1e6
):
    #
    if len(r_map) < 2:
        return np.zeros(len(bins) - 1)

    if not left_bound:
        left_bound = len(r_map)
    else:
        if left_bound > len(r_map):
            print('f')

    site_bins = r_map[:left_bound, np.newaxis] + bins[np.newaxis, :]
    bin_edges = np.searchsorted(r_map, site_bins)
    too_low = bin_edges[:, 0] <= np.arange(left_bound)
    bin_edges[too_low, 0] = np.arange(left_bound)[too_low] + 1
    cum_weights = np.concatenate([[0], np.cumsum(weights)])
    num_pairs = np.zeros(len(bins) - 1, dtype=float)

    for i in np.arange(left_bound):
        if weights[i] > 0:
            num_pairs += weights[i] * np.diff(cum_weights[bin_edges[i]])
            if i % verbosity == 0:
                if i > 0:
                    print(
                        utils.get_time(),
                        f'weighted site pairs parsed at site {i}'
                    )
    return num_pairs


def compute_H2(
    positions,
    genotype_arr,
    vcf_positions,
    r_map,
    bins=None,
    windows=None,
    bounds=None,
    sample_mask=None,
    get_two_sample=True
):
    # across a chromosome
    # num pairs has shape (n_windows, n_bins)
    # num H2 has shape (n_windows, n_samples + n_pairs, n_bins)
    if windows is None:
        windows = np.array([[positions[0], positions[-1] + 1]])
    if bounds is None:
        bounds = np.array([positions[-1] + 1])
    if sample_mask is not None:
        genotype_arr = genotype_arr[:, sample_mask]
    if bins is None:
        bins = _default_bins

    # convert bins in r into bins in cM
    bins = utils.map_function(bins)

    vcf_r_map = r_map[np.searchsorted(positions, vcf_positions)]
    _, n_samples, __ = genotype_arr.shape

    if get_two_sample:
        n_stats = n_samples + utils.n_choose_2(n_samples)
    else:
        n_stats = n_samples

    num_pairs = np.zeros((len(windows), len(bins) - 1))
    num_H2 = np.zeros((len(windows), n_stats, len(bins) - 1))

    for w, (window, bound) in enumerate(zip(windows, bounds)):
        start = np.searchsorted(positions, window[0])
        right_bound = np.searchsorted(positions, bound)
        left_bound = np.searchsorted(positions[start:], window[1])

        num_pairs[w] = count_site_pairs(
            r_map[start:right_bound], bins, left_bound=left_bound
        )

        # these index vcf positions / genotypes
        vcf_start = np.searchsorted(vcf_positions, window[0])
        vcf_right_bound = np.searchsorted(vcf_positions, bound)
        vcf_left_bound = np.searchsorted(vcf_positions[vcf_start:], window[1])

        win_vcf_positions = vcf_positions[vcf_start:vcf_right_bound]
        win_vcf_r_map = vcf_r_map[vcf_start:vcf_right_bound]
        win_genotype_arr = genotype_arr[vcf_start:vcf_right_bound]

        k = 0

        for i in range(n_samples):
            for j in range(i, n_samples):
                if i == j:
                    gts = win_genotype_arr[:, i]
                    site_H = gts[:, 0] != gts[:, 1]
                    H_idx = np.nonzero(site_H)[0]
                    H_r_map = win_vcf_r_map[H_idx]
                    H_positions = win_vcf_positions[H_idx]
                    H_left_bound = np.searchsorted(H_positions, window[1])
                    num_H2[w, k] = count_site_pairs(
                        H_r_map, bins, left_bound=H_left_bound
                    )
                    k += 1
                else:
                    if not get_two_sample:
                        continue
                    gts_i = win_genotype_arr[:, i]
                    gts_j = win_genotype_arr[:, j]
                    site_H = get_two_sample_site_H(gts_i, gts_j)
                    num_H2[w, k] = count_weighted_site_pairs(
                        site_H, win_vcf_r_map, bins, left_bound=vcf_left_bound
                    )
                    k += 1

    return num_pairs, num_H2


def compute_weighted_H2(
    positions,
    genotype_arr,
    vcf_positions,
    r_map,
    weights,
    bins=None,
    windows=None,
    bounds=None,
    sample_mask=None
):

    if windows is None:
        windows = np.array([[positions[0], positions[-1] + 1]])
    if bounds is None:
        bounds = np.array([positions[-1] + 1])
    if sample_mask is not None:
        genotype_arr = genotype_arr[:, sample_mask]
    if bins is None:
        bins = _default_bins

    bins = utils.map_function(bins)

    vcf_r_map = r_map[np.searchsorted(positions, vcf_positions)]
    vcf_weights = weights[np.searchsorted(positions, vcf_positions)]
    _, n_samples, __ = genotype_arr.shape
    norm_constant = np.zeros((len(windows), len(bins) - 1))
    num_H2 = np.zeros(
        (len(windows), utils.n_choose_2(n_samples) + n_samples, len(bins) - 1)
    )

    for z, (window, bound) in enumerate(zip(windows, bounds)):
        start = np.searchsorted(positions, window[0])
        right_bound = np.searchsorted(positions, bounds[z])
        left_bound = np.searchsorted(positions[start:], window[1])

        norm_constant[z] = count_weighted_site_pairs(
            weights[start:right_bound],
            r_map[start:right_bound],
            bins,
            left_bound=left_bound
        )

        vcf_start = np.searchsorted(vcf_positions, window[0])
        vcf_right_bound = np.searchsorted(vcf_positions, bound)
        vcf_left_bound = np.searchsorted(vcf_positions[vcf_start:], window[1])

        win_vcf_positions = vcf_positions[vcf_start:vcf_right_bound]
        win_vcf_r_map = vcf_r_map[vcf_start:vcf_right_bound]
        win_vcf_weights = vcf_weights[vcf_start:vcf_right_bound]
        win_genotype_arr = genotype_arr[vcf_start:vcf_right_bound]

        k = 0

        for i in range(n_samples):
            for j in range(i, n_samples):
                if i == j:
                    gts = win_genotype_arr[:, i]
                    site_H = gts[:, 0] != gts[:, 1]
                    sample_left_bound = np.searchsorted(
                        win_vcf_positions[site_H], window[1]
                    )
                    num_H2[z, k] = count_weighted_site_pairs(
                        win_vcf_weights[site_H],
                        win_vcf_r_map[site_H],
                        bins,
                        left_bound=sample_left_bound
                    )
                else:
                    gts_i = win_genotype_arr[:, i]
                    gts_j = win_genotype_arr[:, j]
                    site_H = get_two_sample_site_H(gts_i, gts_j)
                    pair_weights = site_H * win_vcf_weights
                    num_H2[z, k] = count_weighted_site_pairs(
                        pair_weights,
                        win_vcf_r_map,
                        bins,
                        left_bound=vcf_left_bound
                    )
                k += 1
                
    return norm_constant, num_H2


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
    mask_positions = utils.get_mask_positions(mask_regions)
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
    bins=None,
    get_two_sample=True
):
    #
    if bins is None:
        bins = _default_bins

    t0 = time.time()
    mask_regions = utils.read_mask_file(mask_fname)
    mask_positions = utils.get_mask_positions(mask_regions)
    sample_ids, vcf_positions, genotype_arr = \
        utils.read_vcf_genotypes(vcf_fname, mask_regions)
    r_map = utils.read_map_file(map_fname, mask_positions)
    print(utils.get_time(), 'loaded files')

    num_sites, num_H = compute_H(
        mask_positions,
        genotype_arr,
        vcf_positions,
        windows=windows,
        get_two_sample=get_two_sample
    )
    print(utils.get_time(), 'computed one-locus H')

    num_pairs, num_H2 = compute_H2(
        mask_positions,
        genotype_arr,
        vcf_positions,
        r_map,
        bins=bins,
        windows=windows,
        bounds=bounds,
        get_two_sample=get_two_sample
    )
    print(utils.get_time(), 'computed two-locus H')

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
        f'chromosome {chrom_num} parsed in\t{t} s'
    )
    return stats


def parse_weighted_H2(
    mask_fname,
    vcf_fname,
    map_fname,
    weight_fname,
    bins,
    windows=None,
    bounds=None,
    weight_name='rates',
    inverse_weight=True
):
    #
    if bins is None:
        bins = _default_bins

    t0 = time.time()
    mask_regions = utils.read_mask_file(mask_fname)
    mask_positions = utils.get_mask_positions(mask_regions)
    sample_ids, vcf_positions, genotype_arr = \
        utils.read_vcf_genotypes(vcf_fname, mask_regions)
    r_map = utils.read_map_file(map_fname, mask_positions)
    print(utils.get_time(), "files loaded")

    weight_file = np.load(weight_fname)
    weight_positions = weight_file['positions']
    _weights = weight_file[weight_name]
    weights = _weights[np.searchsorted(weight_positions, mask_positions)]
    if inverse_weight:
        weights = 1 / weights
    print(utils.get_time(), 'scale files loaded')

    # one-locus H
    num_sites, num_H = compute_weighted_H(
        mask_positions,
        genotype_arr,
        vcf_positions,
        weights,
        windows=windows
    )
    print(utils.get_time(), 'one-locus H computed')

    # two-locus H
    num_pairs, num_H2 = compute_weighted_H2(
        mask_positions,
        genotype_arr,
        vcf_positions,
        r_map,
        weights,
        bins=bins,
        windows=windows,
        bounds=bounds
    )
    print(utils.get_time(), 'two-locus H computed')

    n = len(sample_ids)
    stat_ids = [
        (sample_ids[i], sample_ids[j]) for i in np.arange(n) for j in np.arange(i, n)
    ]
    ret = dict(
        n_sites=num_sites,
        H_counts=num_H,
        n_site_pairs=num_pairs,
        H2_counts=num_H2,
        r_bins=bins,
        ids=stat_ids
    )

    t = np.round(time.time() - t0, 0)
    chrom_num = utils.read_vcf_contig(vcf_fname)
    print(
        utils.get_time(),
        f'{len(mask_positions)} sites on '
        f'chromosome {chrom_num} parsed in\t{t} s'
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
