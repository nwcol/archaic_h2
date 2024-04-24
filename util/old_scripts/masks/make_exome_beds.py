
#

import argparse
import numpy as np
from util import masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file_name")
    args = parser.parse_args()
    #
    exome = bed_util.Bed.read_tsv(args.in_file_name)
    for i in np.arange(1, 23):
        chrom_exome = exome.subset_chrom(i)
        chrom_exome.write_bed(f"chr{i}_exome.bed")
