
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


    return 0


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
