
import argparse
import json
import numpy as np
from util import file_util
from util import sample_sets
from util import one_locus
from util import two_locus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vcf_file_name")
    parser.add_argument("bed_file_name")
    parser.add_argument("map_file_name")
    parser.add_argument("window_file_name")
    parser.add_argument("one_sample_file_prefix")
    parser.add_argument("two_sample_file_prefix")
    parser.add_argument("-s", "--sample_ids", nargs='*', default=None)
    parser.add_argument("-t", "--bp_threshold", type=int, default=0)
    parser.add_argument("-r", "--r_bin_file")
    parser.add_argument("-c", "--cross_sample", type=bool, default=True)
    args = parser.parse_args()
    #
    if args.r_bin_file:
        r_edges = np.loadtxt(args.r_bin_file)
    else:
        r_edges = two_locus.r_edges
    with open(args.window_file_name, 'r') as window_file:
        win_dicts = json.load(window_file)["windows"]
    sample_set = sample_sets.SampleSet.read(
        args.vcf_file_name, args.bed_file_name, args.map_file_name
    )
    if args.sample_ids:
        sample_ids = args.sample_ids
    else:
        sample_ids = sample_set.sample_ids
    chrom = sample_set.chrom
    rows = []
    windows = []
    het_counts = {sample_id: [] for sample_id in sample_ids}
    if args.cross_sample:
        sample_pairs = one_locus.enumerate_pairs(sample_ids)
        two_sample_het_counts = {
            sample_pair: [] for sample_pair in sample_pairs
        }
    #
    for win_id in win_dicts:
        win_dict = win_dicts[win_id]
        bounds = win_dict["bounds"]
        lim_right = win_dict["limit_right"]
        rows.append(f"chr{chrom}_win{win_id}")
        windows.append((chrom, bounds))
        # one-sample
        for sample_id in sample_ids:
            het_counts[sample_id].append(
                two_locus.count_het_pairs(
                    sample_set,
                    sample_id,
                    r_edges,
                    window=bounds,
                    limit_right=lim_right,
                    bp_threshold=args.bp_threshold
                )
            )
        # two-sample
        if args.cross_sample:
            for (sample_id_0, sample_id_1) in sample_pairs:
                label = (sample_id_0, sample_id_1)
                two_sample_het_counts[label].append(
                    two_locus.count_two_sample_het_pairs(
                        sample_set,
                        sample_id_0,
                        sample_id_1,
                        r_edges,
                        window=bounds,
                        limit_right=lim_right,
                    )
                )
            #
        print(f"het. site pair counts parsed\twin{win_id}\tchr{chrom}")
    #
    n_r_bins = len(r_edges) - 1
    cols = [f"r_({r_edges[b]}, {r_edges[b + 1]})" for b in range(n_r_bins)]
    #
    for sample_id in sample_ids:
        out_file_name = f"{args.one_sample_file_prefix}_{sample_id}.txt"
        header = dict(
            chrom=chrom,
            statistic="het_pair_counts",
            sample_id=sample_id,
            windows=windows,
            vcf_file=args.vcf_file_name,
            bed_file=args.bed_file_name,
            map_file=args.map_file_name,
            window_file=args.window_file_name,
            cols=cols,
            rows=rows
        )
        file_util.save_arr(
            out_file_name, np.array(het_counts[sample_id]), header=header
        )
    # two-sample
    if args.cross_sample:
        for sample_pair in sample_pairs:
            label = f"{sample_pair[0]},{sample_pair[1]}"
            out_file_name = f"{args.two_sample_file_prefix}_{label}.txt"
            header = dict(
                chrom=chrom,
                statistic="two_sample_het_pair_counts",
                sample_id=sample_pair,
                windows=windows,
                vcf_file=args.vcf_file_name,
                bed_file=args.bed_file_name,
                map_file=args.map_file_name,
                bp_threshold=args.bp_threshold,
                cols=cols,
                rows=rows
            )
            file_util.save_arr(
                out_file_name, np.array(two_sample_het_counts[sample_pair]),
                header=header
            )
    #
    print(f"het. site pair counts parsed on\tchr{chrom}")
