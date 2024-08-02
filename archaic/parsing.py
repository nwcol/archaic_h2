"""
Functions for parsing statistics from .vcf files
"""
import numpy as np
import time

from archaic import masks, one_locus, two_locus, utils


"""
H
"""


def parse_window_H(
    mask_fname,
    vcf_fname,
    windows
):
    mask = masks.Mask.from_bed_file(mask_fname)
    positions = mask.positions
    variant_file = one_locus.VariantFile(vcf_fname, mask=mask)
    genotypes = variant_file.genotypes
    genotype_positions = variant_file.positions
    n_sites, H = one_locus.compute_H(
        genotypes,
        genotype_positions,
        positions,
        windows=windows
    )
    return H / n_sites


"""
H2
"""


def parse_H2(
    mask_fname,
    vcf_fname,
    map_fname,
    windows=None,
    bounds=None,
    r_bins=None
):
    #
    if r_bins is None:
        r_bins = np.logspace(-6, -2, 17)
    t0 = time.time()
    mask = masks.Mask.from_bed_file(mask_fname)
    positions = mask.positions
    variant_file = one_locus.VariantFile(vcf_fname, mask=mask)
    genotypes = variant_file.genotypes
    genotype_positions = variant_file.positions
    r_map = two_locus.get_r_map(map_fname, positions)
    print(utils.get_time(), "files loaded")
    n_sites, H = one_locus.compute_H(
        genotypes,
        genotype_positions,
        positions,
        windows=windows
    )
    n_site_pairs, H2 = two_locus.compute_H2(
        genotypes,
        genotype_positions,
        positions,
        r_map,
        r_bins,
        windows=windows,
        bounds=bounds
    )
    sample_ids = variant_file.sample_ids
    n = len(sample_ids)
    ids = [
        (sample_ids[i], sample_ids[j])
        for i in np.arange(n) for j in np.arange(i, n)
    ]
    stats = dict(
        ids=ids,
        r_bins=r_bins,
        windows=windows,
        bounds=bounds,
        n_sites=n_sites,
        H_counts=H,
        n_site_pairs=n_site_pairs,
        H2_counts=H2
    )
    t = np.round(time.time() - t0, 0)
    chrom_num = variant_file.chrom_num
    print(
        utils.get_time(),
        f'{len(positions)} sites on chromosome {chrom_num} parsed in\t{t} s')
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
