"""
From a .tsv file with columns

ensembl_gene_d ensemble_exon_id chromosome_name exon_chrom_stat exon_chrom_end

write a .bed mask file containing exons for one chromosome. Optionally, 
add a flank of given length to each exonic region.
"""

import argparse
import numpy as np
from archaic import masks
from archaic import utils


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tsv_fname", required=True)
    parser.add_argument("-n", "--chrom_num", type=int, required=True)
    parser.add_argument("-o", "--out_fname", required=True)
    parser.add_argument("-f", "--flank_bp", type=int, default=0)
    return parser.parse_args()


def main():
    #
    args = get_args()
    regions = []
    with open(args.tsv_fname, "rb") as file:
        for i, line in enumerate(file):
            if i > 0:
                _, __, num, start, end = line.decode().split('\t')
                num = int(num)
                if num == args.chrom_num:
                    regions.append([int(start), int(end)])
    regions = np.array(regions)
    if args.flank_bp > 0:
        regions = masks.add_region_flank(regions, args.flank_bp)
    else:
        regions = masks.simplify_regions(regions)
    masks.write_regions(regions, args.out_fname, args.chrom_num)
    n_sites = masks.get_n_sites(regions)
    print(
        utils.get_time(),
        f'exon mask for chrom {args.chrom_num} written; {n_sites} sites'
    )
    return 0


if __name__ == "__main__":
    main()

