
import numpy as np
import sys

from archaic import util


def main():
    #
    fnames = sys.argv[1:]

    print('fname\tn regions\tn sites\tmin size\tmax size')

    tot_reg = 0
    tot_sites = 0
    tot_min = 1e10
    tot_max = 0

    for fname in fnames:
        mask = util.read_mask_file(fname)
        n_reg = len(mask)
        tot_reg += n_reg
        lengths = np.diff(mask)
        n_sites = lengths.sum()
        tot_sites += n_sites
        min_reg = lengths.min()
        tot_min = min(tot_min, min_reg)
        max_reg = lengths.max()
        tot_max = max(tot_max, max_reg)
        print(
            '\t'.join([str(x) for x in [fname, n_reg, n_sites, min_reg, max_reg]])
        )

    print(
        '\t'.join([str(x) for x in ['TOT', tot_reg, tot_sites, tot_min, tot_max]])
    )


if __name__ == '__main__':
    main()
