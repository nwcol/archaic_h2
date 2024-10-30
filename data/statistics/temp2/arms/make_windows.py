import numpy as np
import sys


def main():
    #
    fname = sys.argv[1]
    chrom_nums = np.loadtxt(fname, usecols=(0), dtype=str)
    regions = np.loadtxt(fname, usecols=(1, 2), dtype=int)
    bands = np.loadtxt(fname, usecols=(3), dtype=str)
    bands = np.array([b[0] for b in bands])

    for i in range(1, 23):
        indicator = np.where(chrom_nums == f'chr{i}')[0]
        start = 0
        end = np.max(regions[indicator])
        if i in [13, 14, 15, 21, 22]:
            arr = np.array([[start, end, end]], dtype=int)
        else:
            centromere_idx = np.max(np.where(bands[indicator] == 'p'))
            centromere = regions[indicator][centromere_idx, 1]
            arr = np.array([[start, centromere, centromere], [centromere, end, end]], dtype=int)
        np.savetxt(f'arms_{i}.txt', arr)
    return 0


main()

