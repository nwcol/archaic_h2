"""
Utilities for handling and editing genetic masks
"""


import numpy as np
import gzip


"""
A new class
"""


class Mask(np.ndarray):

    def __new__(
        cls,
        regions,
        dtype=np.int64,
        chrom_num=None
    ):
        #
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
        #
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
        #
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
                chrom_num = b'chr0'
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


"""
Reading .bed and .bed.gz files
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
    return np.array(regions, dtype=np.int64)


def read_mask_positions(mask_fname, first_idx=1):

    regions = read_mask_regions(mask_fname)
    positions = regions_to_positions(regions, first_idx=first_idx)
    return positions


def read_chrom(mask_fname):

    chroms = []
    if ".gz" in mask_fname:
        open_fxn = gzip.open
    else:
        open_fxn = open
    with open_fxn(mask_fname, "rb") as file:
        for line in file:
            chrom, start, stop = line.decode().strip('\n').split('\t')
            if start.isnumeric():
                chroms.append(chrom)
    chroms = np.array(chroms)
    unique = np.unique(chroms)
    if len(unique) != 1:
        print("multiple chromosomes in .bed file")
    return unique


def check_chroms(mask_fnames):
    # make sure all masks have the same chrom column, and return it if so
    chroms = [read_chrom(fname) for fname in mask_fnames]
    unique = np.unique(np.concatenate(chroms))
    if len(unique) != 1:
        print(f"multiple chromosomes in .bed files: {unique}")
    ret = unique[0]
    return ret


"""
Saving masks
"""


def write_regions(regions, out_fname, chrom_num, write_header=False):

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


"""
Reading positions from .vcf files
"""


def read_vcf_positions(vcf_fname):
    # read the column of positions from a .vcf file
    chrom_idx = 0
    pos_idx = 1
    positions = []
    if ".gz" in vcf_fname:
        open_fxn = gzip.open
    else:
        open_fxn = open
    with open_fxn(vcf_fname, "rb") as file:
        for line in file:
            if line.startswith(b'#'):
                continue
            positions.append(line.split(b'\t')[pos_idx])
    chrom_num = line.split(b'\t')[chrom_idx].decode()
    positions = np.array(positions).astype(np.int64)
    return positions, chrom_num


"""
Transformations between region, position, and indicator arrays
"""


def positions_to_indicator(positions, first_idx=1):
    # positions are 1-indexed, indicator is 0-indexed.
    size = positions.max() - first_idx + 1
    indicator = np.zeros(size)
    indicator[positions - first_idx] = 1
    return indicator


def indicator_to_positions(indicator, first_idx=1):

    positions = np.nonzero(indicator)[0] + first_idx
    return positions


def regions_to_indicator(regions):
    # indicator is 0-indexed
    indicator = np.zeros(regions.max())
    for start, stop in regions:
        indicator[start:stop] = True
    return indicator


def indicator_to_regions(indicator):
    # indicator may be either a boolean array or an array of 0s, 1s
    extended = np.concatenate([np.array([0]), indicator, np.array([0])])
    jumps = np.diff(extended)
    starts = np.where(jumps == 1)[0]
    stops = np.where(jumps == -1)[0]
    regions = np.stack([starts, stops], axis=1)
    return regions


def regions_to_positions(regions, first_idx=1):

    indicator = regions_to_indicator(regions)
    positions = indicator_to_positions(indicator, first_idx=first_idx)
    return positions


def positions_to_regions(positions, first_idx=1):

    indicator = positions_to_indicator(positions, first_idx=first_idx)
    regions = indicator_to_regions(indicator)
    return regions


def get_n_sites(regions):
    #
    return np.count_nonzero(regions_to_indicator(regions))


"""
Editing masks
"""


def simplify_regions(regions):
    # merge redundant regions
    indicator = regions_to_indicator(regions)
    _regions = indicator_to_regions(indicator)
    return _regions


def add_region_flank(regions, flank):

    flanks = np.repeat(np.array([[-flank, flank]]), len(regions), axis=0)
    flanked = regions + flanks
    flanked[flanked < 0] = 0
    simplified  = simplify_regions(flanked)
    return simplified


def filter_regions_by_length(regions, min_length):

    lengths = regions[:, 1] - regions[:, 0]
    select = lengths >= min_length
    _regions = regions[select]
    return _regions


"""
Taking unions, intersections etc of multiple masks
"""


def count_overlaps(*region_list):

    lengths = [regions[-1, 1] for regions in region_list]
    max_length = max(lengths)
    overlaps = np.zeros(max_length)
    for i, regions in enumerate(region_list):
        overlaps[:lengths[i]] += regions_to_indicator(regions)
    return overlaps


def intersect_masks(*region_list):
    # take the intersection of regions
    n = len(region_list)
    overlaps = count_overlaps(*region_list)
    indicator = overlaps == n
    regions = indicator_to_regions(indicator)
    return regions


def add_masks(*region_list):
    # take the union of regions
    overlaps = count_overlaps(*region_list)
    indicator = overlaps > 0
    regions = indicator_to_regions(indicator)
    return regions


def subtract_masks(minuend, subtrahend):
    # remove regions in subtrahend from minuend
    lengths = [minuend[-1, 1], subtrahend[-1, 1]]
    max_length = max(lengths)
    mask = np.zeros(max_length)
    mask[:lengths[0]] = regions_to_indicator(minuend)
    mask[:lengths[1]] -= regions_to_indicator(subtrahend)
    regions = indicator_to_regions(mask == 1)
    return regions
