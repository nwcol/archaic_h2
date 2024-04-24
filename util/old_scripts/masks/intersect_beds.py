
#

import argparse
from util import masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_file_name")
    parser.add_argument("bed_file_names", nargs='*')
    parser.add_argument('-m', "--min_region_length", type=int)
    args = parser.parse_args()
    #
    beds = [bed_util.Bed.read_bed(path) for path in args.bed_file_names]
    intersect_bed = bed_util.intersect_beds(*beds)
    if args.min_region_length:
        intersect = intersect_bed.exclude(args.min_region_length)
    intersect_bed.write_bed(args.out_file_name)
