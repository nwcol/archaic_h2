
#

import argparse
import matplotlib.pyplot as plt
from util import plots
from util import sample_sets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vcf_file_name")
    parser.add_argument("bed_file_name")
    parser.add_argument("map_file_name")
    parser.add_argument("window_size", type=int)
    parser.add_argument("out_file_name")
    parser.add_argument("-s", "--sample_ids", nargs='*', default=None)
    parser.add_argument("-y", "--ylim", default=None, type=float)
    args = parser.parse_args()
    #
    sample_set = sample_sets.USampleSet.read(
        args.vcf_file_name, args.bed_file_name, args.map_file_name
    )
    if args.sample_ids:
        sample_ids = args.sample_ids
    else:
        sample_ids = sample_set.sample_ids
    kb = int(args.window_size / 1_000)
    title = f"chr{sample_set.chrom} heterozygosity in {kb}kb bins"
    plot_het = plots.plot_het(
        sample_set, res=args.window_size, title=title, sample_ids=sample_ids,
        ylim=args.ylim
    )
    plt.savefig(args.out_file_name, dpi=200)
