"""
read a .vcf.gz file containing estimates of mutation rate and write:
(1) a .npz archive with a single array, 'rate', with 0 for missing sites
(2) a .bed mask file recording .vcf.gz coverage
"""
import argparse
import gzip
import numpy as np
import sys

from archaic import masks, util


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vcf_fname')
    parser.add_argument('-r', '--out_rate_fname')
    parser.add_argument('-m', '--out_mask_fname')
    return parser.parse_args()


def read(fname, verbose=1e6):

    pos_idx = 1
    info_idx = 7
    mr_idx = 1
    if ".gz" in fname:
        open_fxn = gzip.open
    else:
        open_fxn = open

    # we work in 5Mb blocks to avoid memory issues
    positions = []
    mr = []
    block_positions = []
    block_mr = []

    with open_fxn(fname, "rb") as file:
        for i, line in enumerate(file):
            line = line.decode()
            if line.startswith('#'):
                continue

            if i % 1.5e7 == 0 and i > 0:
                positions.append(block_positions)
                mr.append(block_mr)

            fields = line.strip('\n').split('\t')
            position = int(fields[pos_idx])
            block_positions.append(position)
            info = fields[info_idx].split(';')
            u = float(info[mr_idx].split('=')[1])
            block_mr.append(u)

            if i % (3 * verbose) == 0:
                if i > 0:
                    print(util.get_time(), f'rate read at {i // 3} sites')

    positions = [np.array(pos, dtype=int) for pos in positions]
    mr = [np.array(rate, dtype=float) for rate in mr]

    return positions, mr


def main():

    args = get_args()
    position_blocks, mr_blocks = read(args.vcf_fname)
    rate = np.zeros(position_blocks[-1].max(), dtype=np.float64)
    print(len(position_blocks), len(mr_blocks))
    for i in range(len(position_blocks)):
        sum_rate = np.reshape(mr_blocks[i], (len(mr_blocks[i]) // 3, 3)).sum(1)
        idx = np.unique(position_blocks[i]) - 1
        coeff = 1.015e-7 / 2
        rate[idx] = sum_rate * coeff
    print(util.get_time(), 'rate processed')
    np.savez_compressed(args.out_rate_fname, rate=rate)
    print(util.get_time(), 'rate saved')
    positions = np.concatenate(position_blocks)
    mask = masks.Mask.from_positions(positions)
    print(util.get_time(), 'mask processed')
    mask.write_bed_file(args.out_mask_fname)
    print(util.get_time(), 'mask saved')


if __name__ == '__main__':
    main()
