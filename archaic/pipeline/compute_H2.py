"""
from parsed archives
"""
import argparse
import numpy as np


# field definitions
_ids = 'ids'
_r_bins = 'r_bins'
_n_sites = 'n_sites'
_n_site_pairs = 'n_site_pairs'
_n_H = 'H_counts'
_n_H2 = 'H2_counts'


def get_args():
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--in_fnames', required=True, nargs='*')
    parser.add_argument('-o', '--out_fname', required=True)
    return parser.parse_args()


def main():
    #
    args = get_args()
    file0 = np.load(args.in_fnames[0])
    ids = file0[_ids]
    r_bins = file0[_r_bins]
    for fname in args.in_fnames[1:]:
        file = np.load(fname)
        if np.any(ids != file[_ids]):
            raise ValueError(f'id mismatch in {fname}')
        if np.any(r_bins != file[_r_bins]):
            raise ValueError(f'r bin mismatch in {fname}')
    counts = {_n_sites: 0, _n_site_pairs: 0, _n_H: 0, _n_H2: 0}
    for fname in args.in_fnames:
        file = np.load(fname)
        for stat in [_n_sites, _n_site_pairs, _n_H, _n_H2]:
            counts[stat] += file[stat].sum(0)
    arrs = dict(
        ids=ids,
        r_bins=r_bins,
        H=counts[_n_H] / counts[_n_sites],
        H2=counts[_n_H2] / counts[_n_site_pairs]
    )
    np.savez(args.out_fname, **arrs)


if __name__ == '__main__':
    main()
