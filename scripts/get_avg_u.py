import argparse
import numpy as np

from archaic import utils


def get_args():
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--mask_fnames', nargs='*')
    parser.add_argument('-r', '--rate_fnames', nargs='*')
    return parser.parse_args()


def main():
    #
    args = get_args()
    sum_mu = 0
    num_sites = 0
    for mask_fname, rate_fname in zip(args.mask_fnames, args.rate_fnames):
        mask = utils.read_mask_file(mask_fname)
        bool_mask = utils.get_bool_mask(mask)
        max_idx = len(bool_mask)
        num_sites += bool_mask.sum()
        rate_file = np.load(rate_fname)
        rates = np.zeros(rate_file['positions'][-1] + 1)
        rates[rate_file['positions']] = rate_file['rates']
        sum_mu += rates[:max_idx][bool_mask].sum()
        print(mask_fname, rate_fname)
    print(num_sites, sum_mu, sum_mu / num_sites)
    return 0


if __name__ == '__main__':
    main()
