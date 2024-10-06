"""
compute the mean mutation rate in masked regions in one or more mutation map
files
"""
import argparse
import numpy as np

from archaic import util


def get_args():
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--u_maps', nargs='*', required=True)
    parser.add_argument('-b', '--masks', nargs='*', required=True)
    return parser.parse_args()


def main():
    #
    args = get_args()
    tot_u = 0
    tot_num_sites = 0
    print('chrom\tnum_sites\tavg_u')
    for map_fname, mask_fname in zip(args.u_maps, args.masks):
        reg = util.read_mask_file(mask_fname)
        chrom_num = util.read_mask_chrom_num(mask_fname)
        positions = util.get_mask_positions(reg)
        num_sites = len(positions)
        tot_num_sites += num_sites
        u_map = np.load(map_fname)[positions - 1]
        assert np.all(~np.isnan(u_map))
        u_sum = u_map.sum()
        tot_u += u_sum
        print(
            f'chr{chrom_num}\t{num_sites}\t'
            f'{np.format_float_scientific(u_sum / num_sites, 4)}'
        )
    print(
        f'TOT\t{tot_num_sites}\t'
        f'{np.format_float_scientific(tot_u / tot_num_sites, 4)}\n'
    )
    return 0


if __name__ == '__main__':
    main()
