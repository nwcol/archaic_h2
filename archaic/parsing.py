"""
Functions for parsing statistics from .vcf files
"""
import numpy as np
import time

from archaic import util, counting, dev


_default_bins = np.logspace(-6, -2, 17)
_fine_bins = np.concatenate(
    ([0], np.logspace(-7, -1, 31), np.logspace(-1, -0.3015, 6)[1:])
)


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
    get_two_sample=True
):
    # compute H across a chromosome in one or more windows
    if windows is None:
        windows = np.array([[positions[0], positions[-1] + 1]])

    n_sites = np.diff(np.searchsorted(positions, windows))[:, 0]
    _, n_samples, __ = genotype_arr.shape

    if get_two_sample:
        n_stats = n_samples + util.n_choose_2(n_samples)
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


def compute_SFS(variant_file, ref_as_ancestral=False):
    # variants needs the ancestral allele field in info
    # ref_is_ancestral=True is for simulated data which lacks INFO=AA

    # to-do: break this function up and add windowing so you can compute SFS
    # in bootstrap blocks!!!!
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
        util.get_time(),
        f'{n_triallelic} multiallelic sites, '
        f'{n_mismatch} sites lacking ancestral allele'
    )
    return SFS, sample_ids


"""
computing two-locus statistics
"""


def compute_H2(
    positions,
    genotype_arr,
    genotype_positions,
    r_map,
    bins=None,
    windows=None,
    get_two_sample=True,
    get_denominator=True
):
    # across a chromosome
    # num pairs has shape (n_windows, n_bins)
    # num H2 has shape (n_windows, n_samples + n_pairs, n_bins)
    if windows is None:
        windows = np.array(
            [[positions[0], positions[-1] + 1], positions[-1] + 1]
        )
    if bins is None:
        bins = _default_bins

    # convert bins in r into bins in cM
    bins = util.map_function(bins)

    vcf_r_map = r_map[np.searchsorted(positions, genotype_positions)]
    _, n_samples, __ = genotype_arr.shape

    if get_two_sample:
        n_stats = n_samples + util.n_choose_2(n_samples)
    else:
        n_stats = n_samples

    num_pairs = np.zeros((len(windows), len(bins) - 1))
    num_H2 = np.zeros((len(windows), n_stats, len(bins) - 1))

    for w, (w_start, w_l_end, w_r_end) in enumerate(windows):

        if get_denominator:
            start = np.searchsorted(positions, w_start)
            right_bound = np.searchsorted(positions, w_r_end)
            left_bound = np.searchsorted(positions[start:], w_l_end)

            num_pairs[w] = counting.count_site_pairs(
                r_map[start:right_bound], bins, left_bound=left_bound
            )

        # these index vcf positions / genotypes
        vcf_start = np.searchsorted(genotype_positions, w_start)
        vcf_rbound = np.searchsorted(genotype_positions, w_r_end)
        vcf_lbound = np.searchsorted(genotype_positions[vcf_start:], w_l_end)

        win_vcf_positions = genotype_positions[vcf_start:vcf_rbound]
        win_vcf_r_map = vcf_r_map[vcf_start:vcf_rbound]
        win_genotype_arr = genotype_arr[vcf_start:vcf_rbound]

        k = 0

        for i in range(n_samples):
            for j in range(i, n_samples):
                if i == j:
                    gts = win_genotype_arr[:, i]
                    site_H = gts[:, 0] != gts[:, 1]
                    H_idx = np.nonzero(site_H)[0]
                    H_r_map = win_vcf_r_map[H_idx]
                    H_positions = win_vcf_positions[H_idx]
                    H_lbound = np.searchsorted(H_positions, w_l_end)
                    num_H2[w, k] = counting.count_site_pairs(
                        H_r_map, bins, left_bound=H_lbound
                    )
                    k += 1
                else:
                    if not get_two_sample:
                        continue
                    gts_i = win_genotype_arr[:, i]
                    gts_j = win_genotype_arr[:, j]
                    site_H = get_two_sample_site_H(gts_i, gts_j)
                    num_H2[w, k] = counting.count_weighted_site_pairs(
                        site_H, win_vcf_r_map, bins, left_bound=vcf_lbound
                    )
                    k += 1

        print(util.get_time(), f'computed H2 in window {w}')

    return num_pairs, num_H2


def compute_weighted_H2(
    positions,
    genotype_arr,
    genotype_pos,
    r_map,
    u_map,
    bins=None,
    windows=None,
    get_denominator=True
):

    bins = util.map_function(bins)

    vcf_r_map = r_map[np.searchsorted(positions, genotype_pos)]
    _, n_samples, __ = genotype_arr.shape
    n_stats = util.n_choose_2(n_samples) + n_samples
    num_H2 = np.zeros((len(windows), n_stats, len(bins) - 1))

    """
    if get_denominator:
        denom = counting.compute_chrom_averaged_u_weight(
            positions,
            u_map,
            r_map,
            bins,
            windows
        )
    else:
        denom = np.zeros((len(windows), len(bins) - 1))
    """
    if get_denominator:
        denom = dev.compute_weight_facs(
            positions, r_map, u_map, bins, windows
        )



    for w, (w_start, w_l_end, w_r_end) in enumerate(windows):
        vcf_start = np.searchsorted(genotype_pos, w_start)
        vcf_rbound = np.searchsorted(genotype_pos, w_r_end)
        vcf_lbound = np.searchsorted(genotype_pos[vcf_start:], w_l_end)

        win_vcf_positions = genotype_pos[vcf_start:vcf_rbound]
        win_vcf_r_map = vcf_r_map[vcf_start:vcf_rbound]
        win_genotype_arr = genotype_arr[vcf_start:vcf_rbound]

        k = 0

        for i in range(n_samples):
            for j in range(i, n_samples):
                if i == j:
                    gts = win_genotype_arr[:, i]
                    site_H = gts[:, 0] != gts[:, 1]
                    H_idx = np.nonzero(site_H)[0]
                    H_r_map = win_vcf_r_map[H_idx]
                    H_positions = win_vcf_positions[H_idx]
                    H_lbound = np.searchsorted(H_positions, w_l_end)
                    num_H2[w, k] = dev.count_num_pairs(
                        H_r_map, bins, llim=H_lbound
                    )
                else:
                    gts_i = win_genotype_arr[:, i]
                    gts_j = win_genotype_arr[:, j]
                    site_H = get_two_sample_site_H(gts_i, gts_j)
                    num_H2[w, k] = counting.count_weighted_site_pairs(
                        site_H, win_vcf_r_map, bins, left_bound=vcf_lbound
                    )
                k += 1

        print(util.get_time(), f'computed H2 in window {w}')

    return denom, num_H2


"""
parsing statistics from file
"""


def parse_H(
    mask_fname,
    vcf_fname,
    windows
):
    # from file
    mask_regions = util.read_mask_file(mask_fname)
    mask_positions = util.get_mask_positions(mask_regions)
    sample_ids, vcf_positions, genotype_arr = \
        util.read_vcf_genotypes(vcf_fname, mask_regions)
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
    bins=None,
    get_two_sample=True,
    get_denominator=True
):
    #
    # setup bins
    if isinstance(bins, np.ndarray):
        pass
    elif isinstance(bins, str):
        bins = np.loadtxt(bins, dtype=float)
    else:
        print(util.get_time(), 'parsing with default bins')
        bins = _default_bins

    # setup windows
    if isinstance(windows, np.ndarray):
        if windows.ndim == 1:
            windows = windows[np.newaxis]
        assert windows.shape[1] == 3
    elif isinstance(windows, str):
        windows = np.loadtxt(windows, dtype=int)
        if windows.ndim == 1:
            windows = windows[np.newaxis]
        assert windows.shape[1] == 3
    else:
        windows = None

    t0 = time.time()

    mask_regions = util.read_mask_file(mask_fname)
    mask_positions = util.get_mask_positions(mask_regions)
    sample_ids, vcf_positions, genotype_arr = \
        util.read_vcf_genotypes(vcf_fname, mask_regions)
    r_map = util.read_map_file(map_fname, mask_positions)
    print(util.get_time(), 'loaded files')

    num_sites, num_H = compute_H(
        mask_positions,
        genotype_arr,
        vcf_positions,
        windows=windows[:, :2],
        get_two_sample=get_two_sample
    )
    print(util.get_time(), 'computed one-locus H')

    num_pairs, num_H2 = compute_H2(
        mask_positions,
        genotype_arr,
        vcf_positions,
        r_map,
        bins=bins,
        windows=windows,
        get_two_sample=get_two_sample,
        get_denominator=get_denominator
    )
    print(util.get_time(), 'computed two-locus H')

    n = len(sample_ids)
    if get_two_sample:
        stat_ids = [
            (sample_ids[i], sample_ids[j])
            for i in np.arange(n)
            for j in np.arange(i, n)
        ]
    else:
        stat_ids = np.array(sample_ids)
    stats = dict(
        ids=stat_ids,
        r_bins=bins,
        windows=windows,
        n_sites=num_sites,
        H_counts=num_H,
        n_site_pairs=num_pairs,
        H2_counts=num_H2
    )

    t = np.round(time.time() - t0, 0)
    chrom_num = util.read_vcf_contig(vcf_fname)
    print(
        util.get_time(),
        f'{len(mask_positions)} sites on '
        f'chromosome {chrom_num} parsed in\t{t} s'
    )
    return stats


def parse_weighted_H2(
    mask_fname,
    vcf_fname,
    rmap_fname,
    umap_fname,
    bins=None,
    windows=None,
    get_denominator=True
):
    """

    :param mask_fname:
    :param vcf_fname:
    :param rmap_fname:
    :param umap_fname:
    :param bins:
    :param windows: array with shape (n_windows, 3). the 0th and 1st columns
        hold minimum and maximum positions for left loci and the 2nd column
        holds maximum positions for the right locus. if shape[1] is 2, we
        assume that the maximum position in the windows is the bound
    :return:
    """
    # setup bins
    if isinstance(bins, np.ndarray):
        pass
    elif isinstance(bins, str):
        bins = np.loadtxt(bins, dtype=float)
    else:
        print(util.get_time(), 'parsing with default bins')
        bins = _default_bins

    # setup windows
    if isinstance(windows, np.ndarray):
        if windows.ndim == 1:
            windows = windows[np.newaxis]
        assert windows.shape[1] == 3
    elif isinstance(windows, str):
        windows = np.loadtxt(windows, dtype=int)
        if windows.ndim == 1:
            windows = windows[np.newaxis]
        assert windows.shape[1] == 3
    else:
        windows = None

    t0 = time.time()

    mask_regions = util.read_mask_file(mask_fname)
    mask_positions = util.get_mask_positions(mask_regions)
    sample_ids, vcf_positions, genotype_arr = \
        util.read_vcf_genotypes(vcf_fname, mask_regions)
    r_map = util.read_map_file(rmap_fname, mask_positions)
    print(util.get_time(), "files loaded")

    # load a .npy file
    if umap_fname.endswith('.npy'):
        full_umap = np.load(umap_fname)
        # the umap is 0-indexed
        umap = full_umap[mask_positions - 1]
        if np.any(np.isnan(umap)):
            raise ValueError('nans in mutation map!')
    # load a .bedgraph file containing region-averaged mutation rates
    elif umap_fname.endswith('.bedgraph') or umap_fname.endswith('.bedgraph.gz'):
        regions, data = util.read_bedgraph(umap_fname)
        # assign a mutation rate to each point
        idx = np.searchsorted(regions[:, 1], mask_positions)
        umap = data['u'][idx]
    else:
        raise ValueError(
            'you must provide a .npy or .bedgraph mutation rate file'
        )
    print(util.get_time(), 'weight files loaded')

    # one-locus H
    num_sites, num_H = compute_H(
        mask_positions,
        genotype_arr,
        vcf_positions,
        windows=windows[:, :2]  # one-locus windows
    )
    print(util.get_time(), 'one-locus H computed')

    # two-locus H (H2)
    num_pairs, num_H2 = compute_weighted_H2(
        mask_positions,
        genotype_arr,
        vcf_positions,
        r_map,
        umap,
        bins=bins,
        windows=windows,
        get_denominator=get_denominator
    )
    print(util.get_time(), 'two-locus H computed')

    n = len(sample_ids)
    stat_ids = [
        (sample_ids[i], sample_ids[j])
        for i in np.arange(n)
        for j in np.arange(i, n)
    ]
    ret = dict(
        n_sites=num_sites,
        H_counts=num_H,
        n_site_pairs=num_pairs,
        H2_counts=num_H2,
        r_bins=bins,
        ids=stat_ids,
        windows=windows
    )

    t = np.round(time.time() - t0, 0)
    chrom_num = util.read_vcf_contig(vcf_fname)
    print(
        util.get_time(),
        f'{len(mask_positions)} sites on '
        f'chromosome {chrom_num} parsed in\t{t} s'
    )
    return ret


def bootstrap_H2(
    dics,
    n_iters=1000,
    bin_slice=None
):
    # carry out bootstraps to get H, H2 distributions.
    # takes dictionaries as args
    n_sites = np.concatenate([dic['n_sites'] for dic in dics])
    H_counts = np.concatenate([dic['H_counts'] for dic in dics])
    H2_counts = np.concatenate([dic['H2_counts'] for dic in dics])
    num_pairs = np.concatenate([dic['n_site_pairs'] for dic in dics])

    if bin_slice is None:
        n_windows, n, n_bins = H2_counts.shape
        r_bins = dics[0]["r_bins"]
    else:
        start, stop = bin_slice
        n_windows, n, _ = H2_counts.shape
        r_bins = dics[0]['r_bins'][start:stop + 2]
        print(f'r_bins sliced to {r_bins}')
        n_bins = len(r_bins) - 1
        num_pairs = num_pairs[:, start:stop + 1]
        H2_counts = H2_counts[:, :, start:stop + 1]

    H_dist = np.zeros((n_iters, n))
    H2_dist = np.zeros((n_iters, n, n_bins))
    for i in range(n_iters):
        sample = np.random.randint(n_windows, size=n_windows)
        H_dist[i] = H_counts[sample].sum(0) / n_sites[sample].sum()
        H2_dist[i] = H2_counts[sample].sum(0) / num_pairs[sample].sum(0)

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


def get_mean_H2(*args, fancy=True):
    # take the mean over a number of dictionaries
    dic = args[0]
    ids = dic['ids']
    r_bins = dic['r_bins']
    for dic in args:
        if np.any(ids != dic['ids']):
            raise ValueError(f'id mismatch in {dic}')
        if np.any(r_bins != dic['r_bins']):
            raise ValueError(f'r bin mismatch in {dic}')
        for key in dic:
            if isinstance(dic[key], np.ndarray):
                if dic[key].dtype is int or dic[key].dtype is float:
                    assert not np.any(np.isnan(dic[key]))

    num_pairs = np.vstack([dic['n_site_pairs'] for dic in args])
    if dic['n_sites'].ndim == 0:
        num_sites = np.array([dic['n_sites'] for dic in args])
    else:
        num_sites = np.concatenate([dic['n_sites'] for dic in args])

    num_H2 = np.vstack([dic['H2_counts'] for dic in args])
    num_H = np.vstack([dic['H_counts'] for dic in args])

    mean_H = num_H.sum(0) / num_sites.sum(0)
    mean_H2 = num_H2.sum(0) / num_pairs.sum(0)

    win_H = np.divide(num_H, num_sites[:, np.newaxis], where=num_sites[:, np.newaxis] > 0)
    stderr_H = np.std(win_H, axis=0)

    H2mask = num_pairs[:, np.newaxis] > 0
    win_H2 = np.divide(num_H2, num_pairs[:, np.newaxis], where=H2mask)
    stderr_H2 = np.std(win_H2, axis=0, where=H2mask)

    #if fancy:
        #ret['window_H'] = ret['H_counts'] / ret['n_sites'][:, np.newaxis]
        #ret['window_H2'] = ret['H2_counts'] / ret['n_site_pairs'][:, np.newaxis, :]
        #ret['H'] = ret['H_counts'].sum(0) / ret['n_sites'].sum(0)
        # ret['H2'] = ret['H2_counts'].sum(0) / ret['n_site_pairs'].sum(0)
        #ret['std_H'] = np.std(ret['window_H'], axis=0)
        #ret['std_H2'] = np.std(ret['window_H2'], axis=0)

    ret = dict(
        ids=ids,
        r_bins=r_bins,
        H=mean_H,
        H2=mean_H2,
        std_H=stderr_H,
        std_H2=stderr_H2
    )

    return ret


"""
SFS
"""


def parse_SFS(
    mask_fname,
    vcf_fname,
    out_fname,
    ref_as_ancestral=False):
    #
    """
    n_sites = 0
    mask = masks.read_mask_regions(mask_fname)
    _n_sites = masks.get_n_sites(mask)
    n_sites += _n_sites
    print(
        util.get_time(),
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
            util.get_time(),
            f'{_SFS.sum()} variants parsed to SFS '
            f'on chrom {variants.chrom_num}'
        )
    arrs = dict(
        samples=samples,
        n_sites=n_sites,
        SFS=SFS
    )
    np.savez(out_fname, **arrs)
    """
    return 0


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

    expected_num_pairs = util.n_choose_2(chrom_dict[sites].sum())
    assert num_pairs + chrom_dict[pairs].sum() == expected_num_pairs

    return num_H2, num_pairs


def compute_cross_chrom_H2(chrom_dicts):

    H = 'H_counts'
    sites = 'n_sites'

    n_chroms = len(chrom_dicts)
    n_pairs = util.n_choose_2(n_chroms)
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
