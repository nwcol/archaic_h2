"""

"""

import argparse
from archaic import masks
from archaic import one_locus


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fasta_fname', required=True)
    parser.add_argument('-o', '--out_fname', required=True)
    return parser.parse_args()


def main():

    args = get_args()
    gts, header = one_locus.load_fasta_fmt(args.fasta_fname, simplify=True)
    one_locus.simplify_gts(gts)
    chrom_num = header.split(':')[2]
    regions = one_locus.get_gt_mask(gts)
    masks.write_regions(regions, args.out_fname, chrom_num)
    return 0


if __name__ == '__main__':
    main()
