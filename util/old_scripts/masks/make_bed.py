
#

import argparse
from util import masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vcf_file_name")
    parser.add_argument("bed_file_name")
    args = parser.parse_args()
    #
    mask = bed_util.Bed.read_vcf(args.in_file_name)
    mask.write_bed(args.bed_file_name)
