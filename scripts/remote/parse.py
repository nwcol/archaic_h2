
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

    t0 = time.time()
    mask_regions = masks.read_mask_regions(mask_fname)
    mask_pos = masks.read_mask_positions(mask_fname)
    vcf_pos, samples, genotypes = one_locus.read_vcf_file(
        vcf_fname, mask_regions=mask_regions
    )
    mask_map = two_locus.get_map_vals(map_fname, mask_pos)
    vcf_map = two_locus.get_map_vals(map_fname, vcf_pos)
    sample_names = np.array(samples)
    sample_pairs = one_locus.enumerate_pairs(samples)
    pair_names = np.array([f"{x},{y}" for x, y in sample_pairs])
    print(utils.get_time(), "files loaded")

    ### debug
    print(genotypes)
    print(genotypes.shape)
    print(vcf_pos[one_locus.get_het_idx(genotypes[:, 0])])
    print(vcf_pos[one_locus.get_het_idx(genotypes[:, 1])])

    ###
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


def bootstrap_H2(pair_counts, H2_counts, n_iters=1000):
    #
    n, n_windows, n_bins = H2_counts.shape
    arr = np.zeros((n_iters, n_bins, n))
    for i in range(n_iters):
        idx = np.random.randint(n_windows, size=n_windows)
        H2_sum = H2_counts[:, idx, :].sum(1)
        pair_sum = pair_counts[idx, :].sum(0)
        arr[i] = (H2_sum / pair_sum).T
    return arr


def bootstrap_H(site_counts, H_counts, n_iters=1000):
    #
    n, n_windows = H_counts.shape
    arr = np.zeros((n_iters, n))
    for i in range(n_iters):
        idx = np.random.randint(n_windows, size=n_windows)
        H_sum = H_counts[:, idx].sum(1)
        site_sum = site_counts[idx].sum()
        arr[i] = H_sum / site_sum
    return arr


def bootstrap(in_fnames, out_fname, n_iters=1000):
    #
    in_files = [np.load(x) for x in in_fnames]
    sites = np.concatenate([x["site_counts"] for x in in_files], axis=0)
    H_counts = np.concatenate([x["H_counts"] for x in in_files], axis=1)
    site_pairs = np.concatenate([x["pair_counts"] for x in in_files], axis=0)
    H2_counts = np.concatenate([x["H2_counts"] for x in in_files], axis=1)
    r_bins = in_files[0]["r_bins"]
    n_bins = len(r_bins) - 1
    sample_names = in_files[0]["sample_names"]
    pair_names = in_files[0]["sample_pairs"]
    n = len(sample_names) + len(pair_names)
    H_dist = bootstrap_H(sites, H_counts, n_iters=n_iters)
    H_mean = H_dist.mean(0)
    H_cov = np.cov(H_dist, rowvar=False)
    H2_dist = bootstrap_H2(site_pairs, H2_counts, n_iters)
    H2_cov = np.zeros((n_bins, n, n))
    for i in range(n_bins):
        H2_cov[i, :, :] = np.cov(H2_dist[:, i, :], rowvar=False)
    H2_mean = H2_dist.mean(0)
    arrs = dict(
        sample_names=sample_names,
        pair_names=pair_names,
        r_bins=r_bins,
        H_dist=H_dist,
        H_mean=H_mean,
        H_cov=H_cov,
        H2_dist=H2_dist,
        H2_mean=H2_mean,
        H2_cov=H2_cov
    )
    np.savez(out_fname, **arrs)


"""
SFS
"""


def parse_SFS(in_fnames, out_fname):
    #
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
