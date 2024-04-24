
# remove bins with low coverage from a mask file

import argparse
from util import masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file_name")
    parser.add_argument("out_file_name")
    parser.add_argument('-b', "--bin_size", type=int, default=100_000)
    parser.add_argument('-c', "--coverage", type=float, default=0.50)
    args = parser.parse_args()
    #
    bed = bed_util.Bed.read_bed(args.in_file_name)
    out = bed_util.exclude_low_coverage(bed, args.bin_size, args.coverage)
    out.write_bed(args.out_file_name)
