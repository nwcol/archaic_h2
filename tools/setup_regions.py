"""

"""
import argparse
import numpy as np


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--seq_start',
        type=int, 
        required=True
    )
    parser.add_argument(
        '-l', '--seq_len',
        type=int,
        required=True
    )
    parser.add_argument(
        '-r', '--region_size',
        type=int,
        required=True
    )
    parser.add_argument(
        '-c', '--centromere',
        type=int,
        default=None
    )
    parser.add_argument(
        '-o', '--out_file',
        type=str,
        required=True
    )
    return parser.parse_args()


def for_segment(start, end, size):

    starts = np.arange(start, end, size)
    l_ends = np.append(np.arange(start + size, end, size), end)
    r_ends = np.full(len(l_ends), end)
    regions = np.stack((starts, l_ends, r_ends), axis=1)
    return regions


def main():

    args = get_args()
    seq_start = args.seq_start
    seq_len = args.seq_len
    region_size = args.region_size
    centromere = args.centromere

    if args.centromere is None or centromere == seq_len or centromere == 0:
        regions = for_segment(seq_start, seq_len, region_size)

    else:
        regions_A = for_segment(seq_start, centromere, region_size)
        regions_B = for_segment(centromere, seq_len, region_size)
        regions = np.vstack((regions_A, regions_B))
    
    np.savetxt(args.out_file, regions)
    return 


if __name__ == '__main__':
    main()

