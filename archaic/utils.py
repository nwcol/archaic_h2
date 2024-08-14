"""
Various utilities that are widely used
"""
from datetime import datetime
import gzip
import numpy as np


"""
functions for reading data from file
"""


def read_mask_file(fname):
    #
    regions = np.loadtxt(fname, usecols=(1, 2), dtype=int)
    indicator = np.zeros(regions.max(), dtype=bool)
    for [start, stop] in regions:
        # this is zero-indexed
        indicator[start:stop] = 1
    # this is one-indexed
    positions = np.nonzero(indicator)[0] + 1
    assert not np.any(positions == 0)
    assert len(positions) == np.diff(regions, axis=1).sum()
    return regions, positions


def read_map_file(fname, positions, map_col='Map(cM)'):
    #
    file = open(fname, 'r')
    header = file.readline()
    file.close()
    cols = header.strip('\n').split('\t')
    pos_idx = cols.index('Position(bp)')
    map_idx = cols.index(map_col)
    data = np.loadtxt(fname, skiprows=1, usecols=(pos_idx, map_idx))
    if positions[0] < data[0, 0]:
        print(get_time(), 'positions below map start!')
    if positions[-1] > data[-1, 0]:
        print(get_time(), 'positions above map end!')
    r_map = np.interp(
        positions, data[:, 0], data[:, 1], left=data[0, 1], right=data[-1, 1]
    )
    return r_map


def read_vcf_file(fname, mask_positions=None):
    # returns genotypes
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
            if i == 0:
                gt_index = fields[format_idx].split(':').index('GT')
            positions.append(int(fields[pos_idx]))
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
            genotype_arr.append(genotypes)
            i += 1
    positions = np.array(positions, dtype=int)
    genotype_arr = np.array(genotype_arr, dtype=int)
    if mask_positions is not None:
        idx = np.isin(positions, mask_positions)
        positions = positions[idx]
        genotype_arr = genotype_arr[idx]
    return sample_ids, positions, genotype_arr


"""
recombination map math
"""


def map_function(r):
    # r to cM
    return -50 * np.log(1 - 2 * r)


def inverse_map_func(d):
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


def get_pair_names(items):

    pairs = get_pairs(items)
    pair_names = [f"{x},{y}" for x, y in pairs]
    return pair_names


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
    return int(n * (n - 1) * 0.5)


"""
printouts
"""


def get_time():
    # return a string giving the date and time
    return "[" + datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S") + "]"
