
"""
Utilities for handling and editing genetic masks
"""


import numpy as np
import gzip


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
    return np.array(regions)


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


def save_mask_regions(regions, out_fname, chr_n, write_header=True):

    if ".gz" in out_fname:
        open_fxn = gzip.open
    else:
        open_fxn = open
    with open_fxn(out_fname, "wb") as file:
        if write_header:
            header = "chrom\tchromStart\tchromEnd\n".encode()
            file.write(header)
        for start, stop in regions:
            line = f"{chr_n}\t{start}\t{stop}\n".encode()
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
    chrom = line.split(b'\t')[chrom_idx].decode()
    positions = np.array(positions).astype(np.int64)
    return positions, chrom


def read_vcf_positions_slow(vcf_fname):
    # read the column of positions from a .vcf file. decodes everything first
    pos_idx = 1
    positions = []
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
            positions.append(position)
    positions = np.array(positions)
    return positions
 

"""
Transformations between region, position, and indicator arrays
"""


def positions_to_indicator(positions, first_idx=1):
    # positions assumed to be 1-indexed. bool mask is 0-indexed.
    # might more properly be called an indicator array?
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


"""
Editing masks
"""


def simplify_regions(regions):
    # merge redundant regions
    indicator = regions_to_indicator(regions)
    _regions = indicator_to_regions(indicator)
    return _regions


def add_region_flank(regions, flank):

    flanks = np.repeat([[-flank, flank]], len(regions), axis=0)
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


def get_mask_intersect(*region_list):

    n = len(region_list)
    overlaps = count_overlaps(*region_list)
    indicator = overlaps == n
    regions = indicator_to_regions(indicator)
    return regions


def get_mask_union(*region_list):

    overlaps = count_overlaps(*region_list)
    indicator = overlaps > 0
    regions = indicator_to_regions(indicator)
    return regions


def subtract_masks(minuend, subtrahend):

    lengths = [minuend[-1, 1], subtrahend[-1, 1]]
    max_length = max(lengths)
    mask = np.zeros(max_length)
    mask[:lengths[0]] = regions_to_indicator(minuend)
    mask[:lengths[1]] -= regions_to_indicator(subtrahend)
    regions = indicator_to_regions(mask == 1)
    return regions

