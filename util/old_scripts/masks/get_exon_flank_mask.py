
import argparse
import numpy as np
from util import masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file_name")
    parser.add_argument("exon_file_name")
    parser.add_argument("out_file_name")
    parser.add_argument("-f", "--flank", type=int, default=10_000)
    args = parser.parse_args()
    #
    exon_bed = bed_util.Bed.read_bed(args.exon_file_name)
    flank_bed = bed_util.extend_bed(exon_bed, args.flank)
    in_bed = bed_util.Bed.read_bed(args.in_file_name)
    out_bed = bed_util.subtract_bed(in_bed, flank_bed)
    out_bed.write_bed(args.out_file_name)
