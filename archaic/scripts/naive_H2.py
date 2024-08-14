"""

"""
import argparse
import bisect
import gzip
import numpy as np
import time
from datetime import datetime


def map_function(r):
    # r to cM
    return -50 * np.log(1 - 2 * r)


def get_time():
    return ' [' + datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S") + ']'


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
    first_sample_idx = 9
    positions = []
    genotype_arr = []
    sample_ids = None
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
            positions.append(int(fields[pos_idx]))
            genotypes = []
            for genotype in fields[first_sample_idx:]:
                if '/' in genotype:
                    gt = [int(x) for x in genotype.split('/')]
                elif '|' in genotype:
                    gt = [int(x) for x in genotype.split('|')]
                genotypes.append(gt)
            genotype_arr.append(genotypes)
    positions = np.array(positions, dtype=int)
    genotype_arr = np.array(genotype_arr, dtype=int)
    if mask_positions is not None:
        idx = np.isin(positions, mask_positions)
        positions = positions[idx]
        genotype_arr = genotype_arr[idx]
    return sample_ids, positions, genotype_arr


def count_pairs(r_map, bins, verbosity=1000):

    cM_bins = map_function(bins)
    cM_bins[np.isinf(cM_bins)] = 1000
    n_bins = len(bins) - 1
    num_pairs = np.zeros(n_bins, dtype=np.int64)
    for i, r_i in enumerate(r_map):
        max_idx = bisect.bisect_right(r_map, cM_bins[-1] + r_i)
        print(i, max_idx)
        view = r_map[i + 1:max_idx]
        distances = view - r_i
        for j, dist in enumerate(distances):
            if dist >= bins[0]:
                bin_idx = bisect.bisect_right(cM_bins, dist) - 1
                if bin_idx < n_bins:
                    if bin_idx >= 0:
                        num_pairs[bin_idx] += 1

        if i % verbosity == 0:
            print(get_time(), f'at position {i} of {len(r_map)}')

    return num_pairs


def get_args():
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vcf_fname', required=True)
    parser.add_argument('-r', '--map_fname', required=True)
    parser.add_argument('-b', '--mask_fname', required=True)
    parser.add_argument('-o', '--out_fname', required=True)
    return parser.parse_args()


def main():
    #
    args = get_args()
    mask, mask_positions = read_mask_file(args.mask_fname)
    print(get_time(), 'read mask file')
    r_map = read_map_file(args.map_fname, mask_positions)
    print(get_time(), 'read map file')
    sample_ids, vcf_positions, genotype_arr = \
        read_vcf_file(args.vcf_fname, mask_positions=mask_positions)
    print(get_time(), 'read vcf file')
    vcf_r_map = read_map_file(args.map_fname, vcf_positions)

    r_bins = np.concatenate([[0], np.logspace(-6, -2, 17)])

    num_sites = len(mask_positions)
    num_pairs = count_pairs(r_map, r_bins)
    print(get_time(), 'parsed site pair counts')
    num_H = np.zeros(len(sample_ids))
    num_H2 = np.zeros((len(sample_ids), len(r_bins) - 1))
    for i in range(len(sample_ids)):
        genotypes = genotype_arr[:, i]
        where_het = genotypes[:, 0] != genotypes[:, 1]
        num_H[i] = np.count_nonzero(where_het)
        num_H2[i] = count_pairs(vcf_r_map[where_het], r_bins)
    print(get_time(), 'parsed heterozygous pair counts')
    H = num_H / num_sites
    H2 = num_H2 / num_pairs

    np.savez(
        args.out_fname,
        sample_ids=np.array(sample_ids),
        num_sites=num_sites,
        num_pairs=num_pairs,
        num_H=num_H,
        num_H2=num_H2,
        H=H,
        H2=H2
    )
    print(get_time(), 'saved statistics')
    return 0


if __name__ == '__main__':
    main()
