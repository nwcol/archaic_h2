
#

import argparse
from util import bed_util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bed_file_name")
    parser.add_argument("subtract_file_name")
    parser.add_argument("out_file_name")
    args = parser.parse_args()
    #
    bed = bed_util.Bed.read_bed(args.bed_file_name)
    subtract = bed_util.Bed.read_bed(args.subtract_file_name)
    subset = bed_util.subtract_bed(bed, subtract)
    subset.write_bed(args.out_file_name)
