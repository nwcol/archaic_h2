"""
Functions for computing one-locus statistics.
"""


import gzip
import numpy as np
from archaic import utils
from archaic import masks


"""
Reading .vcf files
"""


def read_vcf_file(vcf_fname, mask_regions=None):
    # read a .vcf or .vcf.gz file and optionally apply a mask to its sites
    # returns vectors of positions, refs, alts, a list of samples, and an
    # array of genotypes with shape (n sites, n samples, 2)
    pos_idx = 1
    ref_idx = 3
    alt_idx = 4
    first_sample_idx = 9
    if np.any(mask_regions):
        # assume 0 indexed
        starts = mask_regions[:, 0]
        stops = mask_regions[:, 1]
        in_mask = lambda x: np.any(np.logical_and(x > starts, x <= stops))
    else:
        in_mask = lambda x: True
    pos = []
    refs = []
    alts = []
    gts = []
    samples = read_vcf_sample_names(vcf_fname)
    n_samples = len(samples)
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
                refs.append(fields[ref_idx])
                alts.append(fields[alt_idx])
    positions = np.array(pos)
    refs = np.array(refs)
    alts = np.array(alts)
    genotypes = np.array(gts)
    return positions, refs, alts, samples, genotypes


def read_vcf_sample_names(vcf_fname):
    # read the sample IDs specified in a .vcf or .vcf.gz file header
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
Manipulating vectors
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
Compute H statistics
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
    # Compute the probabilities that, sampling one allele from x and one from
    # y, they are distinct, for each site.
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
Parsing H from .vcf files
"""


def parse_site_counts(positions, windows):
    # count the number of sites in each window
    # returns array of shape (n_windows)
    site_counts = np.zeros(len(windows), dtype=np.int64)
    for i, window in enumerate(windows):
        site_counts[i] = count_sites(positions, window)
    print(utils.get_time(), "site counts parsed")
    return site_counts


def parse_H_counts(genotypes, positions, windows):
    # count one and two sample H
    # returns array of shape (n_samples + n_pairs, n_windows)
    n_samples = genotypes.shape[1]
    n_pairs = utils.n_choose_2(n_samples)
    n = n_samples + n_pairs
    counts = np.zeros((n, len(windows)), dtype=np.int64)
    for i in range(n_samples):
        for j, window in enumerate(windows):
            counts[i, j] = count_H(
                genotypes[:, i], positions=positions,  window=window
            )
    print(utils.get_time(), "one sample H counts parsed")
    for i, (i_x, i_y) in enumerate(utils.get_pair_idxs(n_samples)):
        i += n_samples
        for j, window in enumerate(windows):
            counts[i, j] = count_Hxy(
                genotypes[:, i_x],
                genotypes[:, i_y],
                positions=positions,
                window=window
            )
    print(utils.get_time(), "two sample H counts parsed")
    return counts


"""
Loading genotypes from .fa or .fasta files
"""


def load_fasta_fmt(fname):
    # expects one sequence per file. returns an array of characters
    if 'gz' in fname:
        open_fxn = gzip.open
    else:
        open_fxn = open
    lines = []
    header = None
    with open_fxn(fname, 'rb') as file:
        for i, line in enumerate(file):
            line = line.decode().rstrip('\n')
            if '>' in line:
                header = line
            else:
                lines.append(line)
    genotypes = np.array(list(''.join(lines)))
    return genotypes, header


def simplify_ancestral_alleles(genotypes):
    #
    symbol_map = {
        'N': '.',
        '-': '.',
        'a': 'A',
        'g': 'G',
        't': 'T',
        'c': 'C'
    }
    for symbol in symbol_map:
        genotypes[genotypes == symbol] = symbol_map[symbol]


def get_genotype_mask(genotypes):
    #
    indicator = genotypes != '.'
    regions = masks.indicator_to_regions(indicator)
    return regions


"""
Computing SFS statistics
"""


def parse_SFS(
    samples,
    genotypes,
    genotype_positions,
    refs,
    alts,
    ancestral_alleles
):
    # exclude triallelic sites
    bi_mask = np.array([',' not in x for x in alts])
    masked_ancestral = ancestral_alleles[genotype_positions[bi_mask] - 1]
    n_excl = len(alts) - bi_mask.sum()
    print(utils.get_time(), f"{n_excl} triallelic sites excluded")
    polarized_genotypes, mask = polarize_genotypes(
        genotypes[bi_mask], refs[bi_mask], alts[bi_mask], masked_ancestral
    )
    _n_excl = bi_mask.sum() - np.sum(mask)
    print(utils.get_time(), f"{_n_excl} non-matching sites excluded")
    derived_counts = polarized_genotypes[mask].sum(2)
    n = len(samples)
    SFS = np.zeros(tuple([3] * n), dtype=np.int64)
    for counts in derived_counts:
        SFS[tuple(counts)] += 1
    return SFS


def polarize_genotypes(genotypes, refs, alts, ancestral_alleles):
    # polarize an array of genotypes so that 0 is ancestral, 1 derived
    # also returns a boolean mask that excludes sites where neither ref nor
    # alt is ancestral
    # triallelic sites aren't allowed and must be masked out before using
    # this function
    ref_matches = refs == ancestral_alleles
    alt_matches = alts == ancestral_alleles
    mask = ref_matches + alt_matches
    polarized_gts = np.zeros(genotypes.shape, dtype=np.int32)
    polarized_gts[ref_matches][genotypes[ref_matches] > 0] = 1
    polarized_gts[alt_matches] = 1 - genotypes[alt_matches]
    return polarized_gts, mask


def two_sample_sfs_matrix(alts):
    # for two samples. i on rows j on cols
    arr = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            arr[i, j] = np.count_nonzero(np.all(alts == [i, j], axis=1))
    arr[0, 0] = 0
    return arr
