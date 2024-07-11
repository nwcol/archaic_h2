"""
Exposes parse functions
"""


import numpy as np
import time
from archaic import masks
from archaic import one_locus
from archaic import two_locus
from archaic import utils


"""
H2
"""


def parse_H2(
    mask_fname,
    vcf_fname,
    map_fname,
    windows,
    bounds,
    r_bins,
    out_fname,
):
    # parse H, H2 stats and save to an .npz archive
    t0 = time.time()
    mask_regions = masks.read_mask_regions(mask_fname)
    mask_pos = masks.read_mask_positions(mask_fname)
    vcf_pos, samples, genotypes = one_locus.read_vcf_file(
        vcf_fname, mask_regions=mask_regions
    )
    mask_map = two_locus.get_map_vals(map_fname, mask_pos)
    vcf_map = two_locus.get_map_vals(map_fname, vcf_pos)
    sample_names = np.array(samples)
    sample_pairs = utils.get_pairs(samples)
    pair_names = np.array([f"{x},{y}" for x, y in sample_pairs])
    print(utils.get_time(), "files loaded")
    site_counts = one_locus.parse_site_counts(mask_pos, windows)
    H_counts = one_locus.parse_H_counts(genotypes, vcf_pos, windows)
    pair_counts = two_locus.parse_pair_counts(
        mask_map, mask_pos, windows, bounds, r_bins
    )
    H2_counts = two_locus.parse_H2_counts(
        genotypes, vcf_map, vcf_pos, windows, bounds, r_bins
    )
    arrs = dict(
        sample_names=sample_names,
        sample_pairs=pair_names,
        windows=windows,
        r_bins=r_bins,
        site_counts=site_counts,
        H_counts=H_counts,
        pair_counts=pair_counts,
        H2_counts=H2_counts
    )
    np.savez(out_fname, **arrs)
    t = np.round(time.time() - t0, 0)
    print(utils.get_time(), f"chromosome parsed in\t{t} s")


def bootstrap_H2(in_fnames, out_fname, n_iters=1000):
    # carry out bootstraps to get H, H2 distributions
    in_files = [np.load(x) for x in in_fnames]
    site_counts = np.concatenate([x["site_counts"] for x in in_files], axis=0)
    H_counts = np.concatenate([x["H_counts"] for x in in_files], axis=1)
    pair_counts = np.concatenate([x["pair_counts"] for x in in_files], axis=0)
    H2_counts = np.concatenate([x["H2_counts"] for x in in_files], axis=1)
    r_bins = in_files[0]["r_bins"]
    sample_names = in_files[0]["sample_names"]
    pair_names = in_files[0]["sample_pairs"]
    n, n_windows, n_bins = H2_counts.shape
    H_dist = np.zeros((n_iters, n))
    H2_dist = np.zeros((n_iters, n_bins, n))
    for i in range(n_iters):
        idx = np.random.randint(n_windows, size=n_windows)
        # H
        H_sum = H_counts[:, idx].sum(1)
        site_sum = site_counts[idx].sum()
        H_dist[i] = H_sum / site_sum
        # H2
        H2_sum = H2_counts[:, idx, :].sum(1)
        pair_sum = pair_counts[idx, :].sum(0)
        H2_dist[i] = (H2_sum / pair_sum).T
    H2_cov = np.zeros((n_bins, n, n))
    for i in range(n_bins):
        H2_cov[i, :, :] = np.cov(H2_dist[:, i, :], rowvar=False)
    arrs = dict(
        sample_names=sample_names,
        pair_names=pair_names,
        r_bins=r_bins,
        H_dist=H_dist,
        H_mean=H_dist.mean(0),
        H_cov=np.cov(H_dist, rowvar=False),
        H2_dist=H2_dist,
        H2_mean=H2_dist.mean(0),
        H2_cov=H2_cov
    )
    np.savez(out_fname, **arrs)


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
        variants = one_locus.Variants(vcf_fname)
        _SFS, _samples = one_locus.parse_SFS(
            variants, ref_as_ancestral=ref_as_ancestral
        )
        if not np.all(_samples == samples):
            raise ValueError(f'sample mismatch: {samples}, {_samples}')
        SFS += _SFS
        print(
            utils.get_time(),
            f'{_SFS.sum()} variants parsed to SFS on chrom {variants.chrom}'
        )
    arrs = dict(
        samples=samples,
        n_sites=n_sites,
        SFS=SFS
    )
    np.savez(out_fname, **arrs)



def parse_two_sample_SFS(in_fnames, out_fname):
    # parse two-sample SFS from .vcf and write SFS arrays to an .npz archive
    # this is an older function and less generalized. also it assumes that all
    # alternate alleles are derived [suitable only for simulations]
    sample_names = None
    genotypes = []
    for fname in in_fnames:
        _, sample_names, gt = one_locus.read_vcf_file(fname)
        genotypes.append(gt)
    genotypes = np.concatenate(genotypes, axis=0)
    alts = genotypes.sum(2)
    n_samples = len(sample_names)
    sfs_arrs = []
    for i, j in utils.get_pair_idxs(n_samples):
        sfs_arrs.append(one_locus.two_sample_sfs_matrix(alts[:, [i, j]]))
    sfs_arr = np.stack(sfs_arrs)
    kwargs = dict(
        sample_names=sample_names,
        pair_names=utils.get_pair_names(sample_names),
        sfs=sfs_arr
    )
    np.savez(out_fname, **kwargs)
