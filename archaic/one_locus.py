"""
Functions for computing one-locus statistics.
"""
import gzip
import numpy as np

from archaic import utils
from archaic import masks





"""
Reading .vcf files in a more naive way
"""


def read_vcf_file(vcf_fname, mask_regions=None):
    # read a .vcf or .vcf.gz file and optionally apply a mask to its sites
    # returns vectors of positions, refs, alts, a list of samples, and an
    # array of genotypes with shape (n sites, n samples, 2)
    pos_idx = 1
    first_sample_idx = 9
    if np.any(mask_regions):
        # assume 0 indexed
        starts = mask_regions[:, 0]
        stops = mask_regions[:, 1]
        in_mask = lambda x: np.any(np.logical_and(x > starts, x <= stops))
    else:
        in_mask = lambda x: True
    positions = []
    genotypes = []
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
                positions.append(position)
                line_gts = []
                for i in range(first_sample_idx, first_sample_idx + n_samples):
                    gt_str = fields[i]
                    if '/' in gt_str:
                        gt = [int(x) for x in fields[i].split('/')]
                    elif '|' in gt_str:
                        gt = [int(x) for x in fields[i].split('|')]
                    line_gts.append(gt)
                genotypes.append(line_gts)
    return np.array(positions), samples, np.array(genotypes)


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
Computing one-locus H from data arrays
"""


def get_one_sample_H(genotypes, genotype_positions, windows):
    # returns counts
    site_H = genotypes[:, 0] != genotypes[:, 1]
    H = np.zeros(len(windows))
    for i, window in enumerate(windows):
        lower, upper = np.searchsorted(genotype_positions, window)
        H[i] = site_H[lower:upper].sum()
    return H


def get_two_sample_H(genotypes_x, genotypes_y, genotype_positions, windows):
    #
    site_H = site_two_sample_H(genotypes_x, genotypes_y)
    H = np.zeros(len(windows))
    for i, window in enumerate(windows):
        lower, upper = np.searchsorted(genotype_positions, window)
        H[i] = site_H[lower:upper].sum()
    return H


def site_two_sample_H(genotypes_x, genotypes_y):
    # compute probabilities of sampling a distinct allele from x and y at each
    H = (
        np.sum(genotypes_x[:, 0][:, np.newaxis] != genotypes_y, axis=1)
        + np.sum(genotypes_x[:, 1][:, np.newaxis] != genotypes_y, axis=1)
    ) / 4
    return H


def compute_H(
    genotypes,
    genotype_positions,
    mask_positions,
    windows=None,
    sample_mask=None,
    verbose=True
):
    #
    if windows is None:
        windows = np.array([[mask_positions[0], mask_positions[-1] + 1]])
    if sample_mask is not None:
        genotypes = genotypes[:, sample_mask]
    n_sites = np.diff(np.searchsorted(mask_positions, windows))[:, 0]
    n_samples = genotypes.shape[1]
    idxs = [(i, j) for i in range(n_samples) for j in np.arange(i, n_samples)]
    H = np.zeros((len(windows), len(idxs)))
    for k, (i, j) in enumerate(idxs):
        if i == j:
            H[:, k] = get_one_sample_H(
                genotypes[:, i], genotype_positions, windows
            )
        else:
            H[:, k] = get_two_sample_H(
                genotypes[:, i], genotypes[:, j], genotype_positions, windows
            )
    if verbose:
        print(
            utils.get_time(),
            f'H parsed for {n_samples} samples '
            f'at {n_sites.sum()} sites in {len(windows)} windows'
        )
    return n_sites, H


"""
Loading genotypes from .fa or .fasta files
"""


def read_fasta_file(fname, map_symbols=True):
    # expects one sequence per file. returns an array of bytes
    if 'gz' in fname:
        open_fxn = gzip.open
    else:
        open_fxn = open
    lines = []
    header = None
    with open_fxn(fname, 'rb') as file:
        for i, line in enumerate(file):
            line = line.rstrip(b'\n')
            if b'>' in line:
                header = line
            else:
                lines.append(line)
    alleles = np.array(list(b''.join(lines).decode()))
    if map_symbols:
        mapping = {'.': 'N', '-': 'N', 'a': 'A', 'g': 'G', 't': 'T', 'c': 'C'}
        for symbol in mapping:
            alleles[alleles == symbol] = mapping[symbol]
    return alleles, header


def get_fa_allele_mask(genotypes):
    # bad name...
    indicator = genotypes != 'N'
    regions = masks.indicator_to_regions(indicator)
    return regions


"""
Computing SFS statistics
"""


def parse_SFS(variant_file, ref_as_ancestral=False):
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
