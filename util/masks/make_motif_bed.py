
import argparse
import numpy as np
from util import bed_util
from util import fasta_util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file_name")
    parser.add_argument("out_file_name")
    parser.add_argument("motif")
    args = parser.parse_args()
    #
    fasta = fasta_util.FastaArr.read_gz(args.in_file_name)
    positions_0 = fasta.get_motif_positions(args.motif)
    chroms = [fasta.chrom]
    bed = bed_util.Bed.from_positions_0(positions_0, chroms)
    bed.write_bed(args.out_file_name)
