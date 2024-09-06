"""
Various utilities that are widely used
"""
from datetime import datetime
import gzip
import numpy as np


"""
recombination map math
"""


def map_function(r):
    # r to cM
    return -50 * np.log(1 - 2 * r)


def inverse_map_function(d):
    # cM to r
    return (1 - np.exp(-d / 50)) / 2


"""
indexing
"""


def get_pairs(items):
    # return a list of 2-tuples containing every pair in 'items'
    n = len(items)
    pairs = []
    for i in np.arange(n):
        for j in np.arange(i + 1, n):
            pair = [items[i], items[j]]
            pair.sort()
            pairs.append((pair[0], pair[1]))
    return pairs


def get_pair_idxs(n):
    # return a list of 2-tuples containing pairs of indices up to n
    pairs = []
    for i in np.arange(n):
        for j in np.arange(i + 1, n):
            pairs.append((i, j))
    return pairs


"""
combinatorics
"""


def n_choose_2(n):
    #
    return int(n * (n - 1) / 2)


"""
printouts
"""


def get_time():
    # return a string giving the date and time
    return "[" + datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S") + "]"



"""
reading and writing data to file
"""


def read_u_bedgraph(fname):
    # read a .bedgraph file with regions in its 1st/2nd col and u in its 4th
    regions = np.loadtxt(fname, usecols=(1, 2), dtype=int)
    u = np.loadtxt(fname, usecols=4, dtype=float)
    edges = np.concatenate(([regions[0, 0]], regions[:, 1]))
    return edges, u


def read_mask_file(fname):
    #
    regions = np.loadtxt(fname, usecols=(1, 2), dtype=int)
    if regions.ndim == 1:
        regions = regions[np.newaxis]
    return regions


def read_mask_chrom_num(fname):
    # return the chromosome number as int, absent any alphabetic characters
    chrom_nums = np.loadtxt(fname, usecols=(0), dtype=str)
    if chrom_nums[0] == 'chrom':
        chrom_nums = chrom_nums[1:]
    unique_nums = np.unique(chrom_nums)
    if len(unique_nums) > 1:
        raise ValueError(f'more than one chromosome is represented in {fname}')
    chrom_num = int(unique_nums[0].lstrip('chrom'))
    return chrom_num


def write_mask_file(regions, out_fname, chrom_num, write_header=False):

    if ".gz" in out_fname:
        open_fxn = gzip.open
    else:
        open_fxn = open
    with open_fxn(out_fname, "wb") as file:
        if write_header:
            header = b'#chrom\tchromStart\tchromEnd\n'
            file.write(header)
        for start, stop in regions:
            line = f'{chrom_num}\t{start}\t{stop}\n'.encode()
            file.write(line)
    return 0


def read_map_file(fname, positions=None, map_col='Map(cM)'):
    #
    file = open(fname, 'r')
    header = file.readline()
    file.close()
    cols = header.strip('\n').split('\t')
    pos_idx = cols.index('Position(bp)')
    map_idx = cols.index(map_col)
    data = np.loadtxt(fname, skiprows=1, usecols=(pos_idx, map_idx))
    if positions is not None:
        if positions[0] < data[0, 0]:
            print(get_time(), 'positions below map start!')
        if positions[-1] > data[-1, 0]:
            print(get_time(), 'positions above map end!')
        r_map = np.interp(
            positions,
            data[:, 0],
            data[:, 1],
            left=data[0, 1],
            right=data[-1, 1]
        )
    else:
        r_map = data[:, 1]
    assert np.all(r_map >= 0)
    assert np.all(np.diff(r_map)) >= 0
    return r_map


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
    regions = get_mask_from_bool(indicator)
    return regions


def read_vcf_genotypes(fname, mask_regions=None, verbosity=1e5):
    # returns genotypes

    if mask_regions is not None:
        bool_mask = get_bool_mask(mask_regions)
        max_position = len(bool_mask)

    def is_in_mask(x):
        #
        if mask_regions is None:
            return True
        else:
            if x < max_position:
                return bool_mask[x]
            else:
                return False

    pos_idx = 1
    format_idx = 8
    first_sample_idx = 9
    positions = []
    genotype_arr = []
    sample_ids = None
    i = 0
    if ".gz" in fname:
        open_fxn = gzip.open
    else:
        open_fxn = open
    with open_fxn(fname, "rb") as file:
        for line_b in file:
            line = line_b.decode()
            if line.startswith('#'):
                if line.startswith('##'):
                    continue
                else:
                    sample_ids = \
                        line.strip('\n').split('\t')[first_sample_idx:]
                    continue

            fields = line.strip('\n').split('\t')
            position = int(fields[pos_idx])

            if i == 0:
                gt_index = fields[format_idx].split(':').index('GT')
            else:
                if i % verbosity == 0:
                    print(get_time(), f'read .vcf row {i}')

            if is_in_mask(position):
                positions.append(position)
                genotypes = []
                for entry in fields[first_sample_idx:]:
                    genotype = entry.split(':')[gt_index]
                    if '/' in genotype:
                        gt = [int(x) for x in genotype.split('/')]
                    elif '|' in genotype:
                        gt = [int(x) for x in genotype.split('|')]
                    else:
                        raise ValueError(r'GT entry has no \ or |')
                    genotypes.append(gt)
                genotype_arr.append(np.array(genotypes))
            i += 1

    print(get_time(), f'finished parsing {i} rows')
    positions = np.array(positions, dtype=int)
    genotype_arr = np.stack(genotype_arr, axis=0, dtype=int)
    print(get_time(), f'finished setting up arrays')
    return sample_ids, positions, genotype_arr


def read_vcf_positions(fname):
    # read and return the vector of positions in a .vcf.gz file
    pos_idx = 1
    positions = []
    if ".gz" in fname:
        open_fxn = gzip.open
    else:
        open_fxn = open
    with open_fxn(fname, "rb") as file:
        for line_b in file:
            if line_b.startswith(b'#'):
                continue
            fields = line_b.strip(b'\n').split(b'\t')
            positions.append(int(fields[pos_idx]))
    return np.array(positions)


def read_vcf_sample_ids(vcf_fname):
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


def read_vcf_rates(
    fname,
    rate_tag='MR',
    verbosity=1e6
):
    # reads mutation rates from a Roulette .vcf.gz file
    verbosity *= 3
    pos_idx = 1
    info_idx = 7
    rate_idx = None
    positions = []
    rates = []
    last_position = None
    i = 0
    with gzip.open(fname, 'rb') as file:
        for line_b in file:
            line = line_b.decode()
            if line[0] == '#':
                continue

            if i == 0:
                _fields = line.strip('\n').split('\t')
                _info = _fields[info_idx].split(';')
                names = [x.split('=')[0] for x in _info]
                if rate_tag not in names:
                    raise ValueError(f'tag {rate_tag} not present in info!')
                rate_idx = names.index(rate_tag)

            fields = line.strip('\n').split('\t')
            position = int(fields[pos_idx])
            info = fields[info_idx].split(';')
            rate = float(info[rate_idx].split('=')[1])

            if position == last_position:
                rates[-1] += rate
            else:
                positions.append(position)
                rates.append(rate)

            last_position = position
            i += 1
            if i % verbosity == 0:
                if i > 0:
                    n = i // 3
                    print(get_time(), f'rate parsed for {n} sites')
    print(get_time(), f'read rates for {len(positions)} positions from .vcf')
    positions = np.array(positions)
    rates = np.array(rates)
    print(get_time(), 'set up position and rate arrays')
    return positions, rates


def read_vcf_contig(fname):
    # return the contig id of the first row of a .vcf.gz file
    if ".gz" in fname:
        open_func = gzip.open
    else:
        open_func = open

    chrom_idx = 0

    with open_func(fname, 'rb') as file:
        for line_b in file:
            line = line_b.decode()
            if line[0] == '#':
                continue
            else:
                chrom = line.strip('\n').split('\t')[chrom_idx]
                break
    return chrom


"""
manipulating masks and transforming them into various forms
"""


def get_bool_mask(mask):
    # turn an array of mask regions into a 1-indexed boolean mask
    # e.g. the zeroth element of the mask always equals False
    bool_mask = np.zeros(mask.max() + 1, dtype=bool)
    for (start, end) in mask:
        bool_mask[start + 1:end + 1] = True
    assert bool_mask[0] == False
    assert bool_mask[-1] == True
    return bool_mask


def get_bool_mask_0(regions):
    # get a 0-indexed boolean mask representing mask coverage
    bool_mask = np.zeros(regions.max())
    for start, end in regions:
        bool_mask[start:end] = True
    assert bool_mask[-1] == True
    return bool_mask


def get_mask_positions(regions):
    # get a vector of 1-indexed positions from an array of mask regions
    positions = np.nonzero(get_bool_mask(regions))[0]
    assert positions[0] >= 1
    return positions


def get_mask_from_bool(bool_mask):
    # construct an array of mask regions from a 0-indexed boolean mask
    _bool_mask = np.concatenate([[0], bool_mask, [0]])
    jumps = np.diff(_bool_mask)
    starts = np.where(jumps == 1)[0]
    stops = np.where(jumps == -1)[0]
    regions = np.stack([starts, stops], axis=1)
    assert np.all(regions > 0)
    return regions


def collapse_mask(mask):
    # merge redundant regions together
    bool_mask = get_bool_mask_0(mask)
    ret = get_mask_from_bool(bool_mask)
    return ret


def add_mask_flank(mask, flank):
    # extend each mask region 'flank' bp in each direction
    flanks = np.repeat(np.array([[-flank, flank]]), len(mask), axis=0)
    flanked = mask + flanks
    flanked[flanked < 0] = 0
    ret = collapse_mask(flanked)
    return ret


def filter_mask_by_length(mask, min_length):
    # exclude regions smaller than min_length
    lengths = mask[:, 1] - mask[:, 0]
    select = lengths >= min_length
    ret = mask[select]
    return ret


def count_mask_overlaps(*masks):

    lengths = [mask[-1, 1] for mask in masks]
    max_length = max(lengths)
    overlaps = np.zeros(max_length)
    for i, regions in enumerate(masks):
        overlaps[:lengths[i]] += get_bool_mask_0(regions)
    return overlaps


def intersect_masks(*masks):
    # take the intersection of mask regions
    n = len(masks)
    overlaps = count_mask_overlaps(*masks)
    indicator = overlaps == n
    regions = get_mask_from_bool(indicator)
    return regions


def add_masks(*masks):
    # take the union of mask regions
    overlaps = count_mask_overlaps(*masks)
    indicator = overlaps > 0
    regions = get_mask_from_bool(indicator)
    return regions


def subtract_masks(minuend, subtrahend):
    # remove regions in subtrahend from minuend
    lengths = [minuend[-1, 1], subtrahend[-1, 1]]
    max_length = max(lengths)
    bool_mask = np.zeros(max_length)
    bool_mask[:lengths[0]] = get_bool_mask_0(minuend)
    bool_mask[:lengths[1]] -= get_bool_mask_0(subtrahend)
    regions = get_mask_from_bool(bool_mask == 1)
    return regions


"""
a class for rapidly reading and accessing small .vcf files
"""


class VariantFile:
    """
    loads a .vcf or .vcf.gz file and holds its contents in memory as a numpy
    byte array. allows rapid access to various fields
    """

    # vcf column indices
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
a class for loading/manipulating .bed mask files
"""


class Mask(np.ndarray):

    def __new__(
        cls,
        regions,
        dtype=np.int64,
        chrom_num=None
    ):
        """
        loads a .bed mask file as an array

        :param regions: 2dim numpy array of shape (n_regions, 2)
        :param dtype: optional. defines array data type
        :param chrom_num: optional. tracks chromosome number. should be numeric
            (e.g. a bed file with chromosome chr8 has chrom_num=8)
        :dtype chrom_num: int or str.
        """
        arr = np.asanyarray(regions, dtype=dtype).view(cls)
        arr.chrom_num = chrom_num

        # array must have ndim 2
        if arr.ndim != 2:
            raise ValueError(
                'regions must have regions.ndim == 2'
            )

        # array must have arr.shape[1] = 2
        if arr.shape[1] != 2:
            raise ValueError(
                'regions must have regions.shape[1] == 2'
            )
        return arr

    def __array_finalize__(self, obj):

        if obj is None:
            return
        np.ndarray.__array_finalize__(self, obj)
        self.chrom_num = getattr(obj, 'chrom_num', None)

    @classmethod
    def from_bed_file(cls, fname):

        regions = []
        if ".gz" in fname:
            open_func = gzip.open
        else:
            open_func = open
        with open_func(fname, "rb") as file:
            for line in file:
                chrom, start, stop = line.decode().strip('\n').split('\t')
                if start.isnumeric():
                    regions.append([int(start), int(stop)])
        regions = np.array(regions, dtype=np.int64)
        chrom_num = int(chrom.lstrip('chr'))
        return cls(regions, chrom_num=chrom_num)

    @classmethod
    def from_vcf_file(cls, fname):
        """
        read a mask of .vcf position coverage

        :param fname: path to .vcf or vcf.gz file
        :return: class instance
        """
        chrom_idx = 0
        pos_idx = 1
        positions = []
        if ".gz" in fname:
            open_func = gzip.open
        else:
            open_func = open
        with open_func(fname, "rb") as file:
            for line in file:
                if line.startswith(b'#'):
                    continue
                positions.append(line.split(b'\t')[pos_idx])
        chrom_num = line.split(b'\t')[chrom_idx].decode()
        positions = np.array(positions).astype(np.int64)
        regions = cls.positions_to_regions(positions)
        return cls(regions, chrom_num=chrom_num)

    @classmethod
    def from_boolean(cls, boolean_mask, chrom_num=None):
        #
        regions = cls.boolean_to_regions(boolean_mask)
        return cls(regions, chrom_num=chrom_num)

    @classmethod
    def from_positions(cls, positions, chrom_num=None):
        #
        regions = cls.positions_to_regions(positions)
        return cls(regions, chrom_num=chrom_num)

    @property
    def boolean(self):
        # 0-indexed
        boolean_mask = np.zeros(self.max(), dtype=bool)
        for start, stop in self:
            boolean_mask[start:stop] = True
        return boolean_mask

    @property
    def positions(self):
        # 1-indexed
        return np.nonzero(self.boolean)[0] + 1

    @property
    def n_sites(self):
        #
        return self.boolean.sum()

    @classmethod
    def positions_to_regions(cls, positions):
        #
        return cls.boolean_to_regions(cls.positions_to_boolean(positions))

    @staticmethod
    def boolean_to_regions(boolean):
        #
        _boolean = np.concatenate([np.array([0]), boolean, np.array([0])])
        jumps = np.diff(_boolean)
        regions = np.stack(
            [np.where(jumps == 1)[0], np.where(jumps == -1)[0]], axis=1
        )
        return regions

    @staticmethod
    def positions_to_boolean(positions):
        #
        boolean_mask = np.zeros(positions.max(), dtype=bool)
        boolean_mask[positions - 1] = True
        return boolean_mask

    def write_bed_file(self, fname, write_header=False, chrom_num=None):
        #
        if chrom_num is None:
            if self.chrom_num is None:
                chrom_num = 'chr0'
            else:
                chrom_num = ('chr' + str(self.chrom_num)).encode()
        if ".gz" in fname:
            open_fxn = gzip.open
        else:
            open_fxn = open
        with open_fxn(fname, "wb") as file:
            if write_header:
                header = b'#chrom\tchromStart\tchromEnd\n'
                file.write(header)
            for start, stop in self:
                line = f'{chrom_num}\t{start}\t{stop}\n'.encode()
                file.write(line)
        return 0


class TwoLocusMask(np.ndarray):

    def __new__(cls, regions):
        pass
