"""
Functions for computing one-locus statistics.
"""


import gzip
import numpy as np
from archaic import utils
from archaic import masks


"""
Experimental class for reading .vcf.gz files
"""

# rename to VariantFile later


class Variants:
    # loads a .vcf file and holds its contents in memory

    chrom_idx = 0
    pos_idx = 1
    ref_idx = 3
    alt_idx = 4
    qual_idx = 5
    filter_idx = 6
    info_idx = 7
    format_idx = 8
    sample_0_idx = 9

    def __init__(self, vcf_fname, mask=None):
        #
        if ".gz" in vcf_fname:
            open_fxn = gzip.open
        else:
            open_fxn = open
        _meta_info = []
        _lines = []
        with open_fxn(vcf_fname, "rb") as file:
            for line in file:
                if line.startswith(b'##'):
                    _meta_info.append(line.strip(b'\n'))
                else:
                    _lines.append(line.strip(b'\n'))
        self.meta_info = np.array(_meta_info)
        self.header = _lines[0]
        lines = np.array(_lines[1:])
        _positions = np.array(
            [line.split(b'\t')[self.pos_idx] for line in lines], dtype=np.int64
        )
        if mask is not None:
            boolean_mask = mask.boolean
            bool_mask = np.zeros(len(lines), dtype=bool)
            in_mask = _positions <= len(boolean_mask)
            bool_mask[in_mask] = boolean_mask[_positions[in_mask] - 1] == 1
        else:
            bool_mask = np.full(len(lines), True)
        self.lines = lines[bool_mask]
        self.positions = _positions[bool_mask]

    def __len__(self):
        #
        return len(self.lines)

    @property
    def sample_ids(self):
        # shape (len(sample_ids))
        return np.array(self.header.decode().split('\t')[self.sample_0_idx:])

    @property
    def genotypes(self):
        # shape (len(positions), len(samples), 2)
        genotypes = []
        for line in self.lines:
            fields = line.split(b'\t')
            idx = fields[self.format_idx].split(b'\t').index(b'GT')
            line_genotypes = [
                gt.split(b':')[idx].split(b'/') if b'/' in gt
                else gt.split(b':')[idx].split(b'|')
                for gt in fields[self.sample_0_idx:]
            ]
            genotypes.append(line_genotypes)
        return np.array(genotypes, dtype=np.int32)

    @property
    def fast_genotypes(self):
        # assumes that GT is the only format field and data is unphased
        # only marginally faster...
        genotypes = []
        for line in self.lines:
            fields = line.split(b'\t')
            line_genotypes = [
                gt.split(b'/') for gt in fields[self.sample_0_idx:]
            ]
            genotypes.append(line_genotypes)
        return np.array(genotypes, dtype=np.int32)

    @property
    def refs(self):
        # returned as bytes
        refs = [line.split(b'\t')[self.ref_idx] for line in self.lines]
        return np.array(refs, dtype=np.str_)

    @property
    def alts(self):
        # returned as bytes. returns multiallelic alts comma-separated
        alts = [line.split(b'\t')[self.alt_idx] for line in self.lines]
        return np.array(alts, dtype=np.str_)

    @property
    def ancestral_alleles(self):
        #
        ancestral_alleles = []
        for line in self.lines:
            info = line.split(b'\t')[self.info_idx]
            info_dict = self.get_info_dict(info)
            if b'AA' in info_dict:
                ancestral_alleles.append(info_dict[b'AA'])
            else:
                ancestral_alleles.append(b'N')
        return np.array(ancestral_alleles, dtype=np.str_)

    @property
    def chrom_num(self):
        #
        return self.lines[0].split(b'\t')[self.chrom_idx].decode()

    @staticmethod
    def get_info_dict(info):
        # turns b'key=value;...' into {key: value, ...}
        if info != b'.':
            nested = [x.split(b'=') for x in info.split(b';')]
            info_dict = {x: y for x, y in nested}
        else:
            info_dict = {}
        return info_dict

    def access_info(self, field):
        #

        return 0

    def access_format(self, field):
        #

        return 0


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
Manipulating vectors
"""


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

def get_window_bounds(window, positions):

    return np.searchsorted(positions, window)


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
New
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
    site_H = get_site_two_sample_H(genotypes_x, genotypes_y)
    H = np.zeros(len(windows))
    for i, window in enumerate(windows):
        lower, upper = np.searchsorted(genotype_positions, window)
        H[i] = site_H[lower:upper].sum()
    return H


def get_site_two_sample_H(genotypes_x, genotypes_y):
    # compute probabilities of sampling a distinct allele from x and y at each
    site_two_sample_H = (
        np.sum(genotypes_x[:, 0][:, np.newaxis] != genotypes_y, axis=1)
        + np.sum(genotypes_x[:, 1][:, np.newaxis] != genotypes_y, axis=1)
    ) / 4
    return site_two_sample_H


def compute_H(
    genotypes,
    genotype_positions,
    mask_positions,
    windows=None,
    sample_mask=None,
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
    print(
        utils.get_time(),
        f'H parsed for {n_samples} samples '
        f'at {n_sites} sites in {len(windows)} windows'
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


def parse_SFS(variants, ref_as_ancestral=False):
    # variants needs the ancestral allele field in info
    # ref_is_ancestral=True is for simulated data which lacks INFO=AA
    genotypes = variants.genotypes
    refs = variants.refs
    alts = variants.alts
    if ref_as_ancestral:
        ancs = refs
    else:
        ancs = variants.ancestral_alleles
    sample_ids = variants.sample_ids
    n = len(sample_ids)
    SFS = np.zeros([3] * n, dtype=np.int64)
    n_triallelic = 0
    n_mismatch = 0
    for i in range(len(variants)):
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
