
import gzip
import numpy as np


"""
Read files
"""


def read_mask_regions(mask_fname):

    regions = []
    if ".gz" in mask_fname:
        open_fxn = gzip.open
    else:
        open_fxn = open
    with open_fxn(mask_fname, "rb") as file:
        for line in file:
            _, start, stop = line.decode().strip('\n').split('\t')
            if start.isnumeric():
                regions.append([int(start), int(stop)])
    return np.array(regions)


def mask_positions_from_regions(regions, first_idx=1):

    bool_mask = np.zeros(regions.max())
    for start, stop in regions:
        bool_mask[start:stop] = True
    positions = np.nonzero(bool_mask)[0]
    positions += first_idx
    return positions


def read_mask_positions(mask_fname, first_idx=1):

    regions = read_mask_regions(mask_fname)
    bool_mask = np.zeros(regions.max())
    for [start, stop] in regions:
        bool_mask[start:stop] = True
    positions = np.nonzero(bool_mask)[0]
    positions += first_idx
    return positions


def read_vcf_file(vcf_fname, mask_regions=None):

    pos_idx = 1
    first_sample_idx = 9

    if np.any(mask_regions):
        # assume 0 indexed
        starts = mask_regions[:, 0]
        stops = mask_regions[:, 1]
        in_mask = lambda x: np.any(np.logical_and(x > starts, x <= stops))
    else:
        in_mask = lambda x: True
    pos = []
    gts = []
    sample_names = read_vcf_sample_names(vcf_fname)
    n_samples = len(sample_names)
    if ".gz" in vcf_fname:
        open_fxn = gzip.open
    else:
        open_fxn = open
    with open_fxn(vcf_fname, "rb") as file:
        for line_b in file:
            line = line_b.decode()
            if line.startswith('#'):
                continue
            fields = line.strip('\n').split('\t')
            position = int(fields[pos_idx])
            if in_mask(position):
                pos.append(position)
                line_gts = []
                for i in range(first_sample_idx, first_sample_idx + n_samples):
                    gt_str = fields[i]
                    if '/' in gt_str:
                        gt = [int(x) for x in fields[i].split('/')]
                    elif '|' in gt_str:
                        gt = [int(x) for x in fields[i].split('|')]
                    line_gts.append(gt)
                gts.append(line_gts)
    positions = np.array(pos)
    genotypes = np.array(gts)
    return positions, sample_names, genotypes


def read_vcf_sample_names(vcf_fname):

    first_sample_idx = 9

    if ".gz" in vcf_fname:
        open_fxn = gzip.open
    else:
        open_fxn = open
    with open_fxn(vcf_fname, "rb") as file:
        for line_b in file:
            line = line_b.decode()
            if line.startswith('#CHROM'):
                break
    fields = line.strip('\n').split('\t')
    sample_names = fields[first_sample_idx:]
    return sample_names


"""
Manipulate vectors
"""


def get_window_bounds(window, positions):

    return np.searchsorted(positions, window)


def slice_window(positions, window, *args):

    start, stop = get_window_bounds(window, positions)
    out_positions = positions[start:stop]
    out_args = [arg[start:stop] for arg in args]
    if len(out_args) == 0:
        ret = out_positions
    else:
        ret = [out_positions] + out_args
    return ret


def get_alt_counts(genotypes):

    alt_counts = np.sum(genotypes != 0, axis=1)
    return alt_counts


"""
Compute statistics
"""


def count_sites(positions, window=None):

    if np.any(window):
        n_sites = len(slice_window(positions, window))
    else:
        n_sites = len(positions)
    return n_sites


def get_site_H(genotypes):

    return genotypes[:, 0] != genotypes[:, 1]


def get_het_idx(genotypes):

    return np.nonzero(get_site_H(genotypes))


def get_het_sites(genotypes, positions):

    return positions[get_het_idx(genotypes)]


def count_H(genotypes, positions=None, window=None):

    if np.any(window):
        if np.any(positions):
            if len(positions) != len(genotypes):
                raise ValueError("Genotype/position length mismatch")
            positions, genotypes = slice_window(
                positions, window, genotypes
            )
        else:
            raise ValueError("You must provide positions to use a window!")
    H = np.count_nonzero(get_site_H(genotypes))
    return H


def get_site_Hxy(genotypes_x, genotypes_y):
    """
    Compute the probabilities that, sampling one allele from x and one from y,
    they are distinct, at each site.

    :param genotypes_x:
    :param genotypes_y:
    :return:
    """
    site_Hxy = (
            np.sum(genotypes_x[:, 0][:, np.newaxis] != genotypes_y, axis=1)
            + np.sum(genotypes_x[:, 1][:, np.newaxis] != genotypes_y, axis=1)
    ) / 4
    return site_Hxy


def count_Hxy(genotypes_x, genotypes_y, positions=None, window=None):

    if np.any(window):
        if np.any(positions):
            if len(positions) != len(genotypes_x):
                raise ValueError("Genotype/position length mismatch")
            if len(genotypes_x) != len(genotypes_y):
                raise ValueError("Genotype/genotype length mismatch")
            positions, genotypes_x, genotypes_y = slice_window(
                positions, window, genotypes_x, genotypes_y
            )
        else:
            raise ValueError("You must provide positions to use a window!")
    Hxy = get_site_Hxy(genotypes_x, genotypes_y).sum()
    return Hxy


def count_approx_Hxy(genotypes_x, genotypes_y, positions=None, window=None):
    """
    Approximate in that it handles all alternate alleles alike and will treat
    sites heterozygous for two different alternate alleles as homozygous

    :param genotypes_x:
    :param genotypes_y:
    :param positions:
    :param window:
    :return:
    """
    if np.any(window):
        if np.any(positions):
            if len(positions) != len(genotypes_x):
                raise ValueError("Genotype/position length mismatch")
            if len(genotypes_x) != len(genotypes_y):
                raise ValueError("Genotype/genotype length mismatch")
            positions, genotypes_x, genotypes_y = slice_window(
                positions, window, genotypes_x, genotypes_y
            )
        else:
            raise ValueError("You must provide positions to use a window!")
    alt_counts_x = get_alt_counts(genotypes_x)
    alt_counts_y = get_alt_counts(genotypes_y)
    Hxy = np.sum(
        (2 - alt_counts_x) * alt_counts_y + (2 - alt_counts_y) * alt_counts_x
    ) / 4
    return Hxy


"""
Utilities
"""


def enumerate_pairs(items):

    n = len(items)
    pairs = []
    for i in np.arange(n):
        for j in np.arange(i + 1, n):
            pair = [items[i], items[j]]
            pair.sort()
            pairs.append((pair[0], pair[1]))
    return pairs


def enumerate_indices(n):

    pairs = []
    for i in np.arange(n):
        for j in np.arange(i + 1, n):
            pairs.append((i, j))
    return pairs
