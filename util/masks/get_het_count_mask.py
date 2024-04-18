
import argparse
import numpy as np
from util import bed_util
from util import sample_sets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chrom")
    parser.add_argument("out_file_name")
    parser.add_argument("-r", "--resolution", type=int, default=10_000)
    parser.add_argument("-t", "--threshold", type=int, default=50)
    args = parser.parse_args()
    #
    sample_set = sample_sets.SampleSet.read_chr(args.chrom)
    sample_ids = sample_set.sample_ids
    n_bins = sample_set.last_position // args.resolution + 1
    bins = np.arange(
        0, (n_bins + 1) * args.resolution, args.resolution, dtype=np.int64
    )
    tally = np.zeros(n_bins)
    for sample_id in sample_ids:
        het_sites = sample_set.het_sites(sample_id)
        counts, _bins = np.histogram(het_sites, bins=bins)
        tally[counts > args.threshold] += 1
    mask = tally > 0
    regions = np.zeros((np.sum(mask), 2), dtype=np.int64)
    regions[:, 0] = bins[:-1][mask]
    regions[:, 1] = bins[1:][mask]
    chrom = np.full(len(regions), sample_set.chrom)
    bed = bed_util.Bed(regions, chrom)
    bed.write_bed(args.out_file_name)
