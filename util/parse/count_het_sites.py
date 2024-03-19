
import argparse
import json
from util import file_util
from util import sample_sets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vcf_file_name")
    parser.add_argument("bed_file_name")
    parser.add_argument("map_file_name")
    parser.add_argument("window_file")
    parser.add_argument("out_file_prefix")
    parser.add_argument("-s", "--sample_ids", nargs='*', default=None)
    args = parser.parse_args()
    #
    with open(args.window_file, 'r') as window_file:
        window_dicts = json.load(window_file)["windows"]
    sample_set = sample_sets.USampleSet.read(
        args.vcf_file_name, args.bed_file_name, args.map_file_name
    )
    chrom = sample_set.chrom
    if args.sample_ids:
        sample_ids = args.sample_ids
    else:
        sample_ids = sample_set.sample_ids
    het_counts = {sample_id: {} for sample_id in sample_ids}
    #
    for window_id in window_dicts:
        window_dict = window_dicts[window_id]
        bounds = window_dict["bounds"]
        row_name = f"chr{chrom}_win{window_id}"
        for sample_id in sample_ids:
            hets = sample_set.count_het_sites(sample_id, bounds=bounds)
            het_counts[sample_id][row_name] = hets
        print(f"HET SITES COUNTED IN\tWIN{window_id}\tCHR{chrom}")
    #
    for sample_id in sample_ids:
        out_file_name = f"{args.out_file_prefix}_{sample_id}.txt"
        header = file_util.get_header(
            chrom=chrom,
            statistic="het_site_counts",
            sample_id=sample_id,
            windows=window_dicts,
            vcf_file=args.vcf_file_name,
            bed_file=args.bed_file_name,
            map_file=args.map_file_name,
            cols={0: "het_site_counts"}
        )
        file_util.save_dict_as_arr(
            out_file_name, het_counts[sample_id], header=header
        )
    #
    print(f"HET SITES COUNTED ON\tCHR{chrom}")
