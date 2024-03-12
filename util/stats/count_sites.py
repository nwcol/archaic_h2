
import argparse
import json
import numpy as np
from util import bed_util
from util import file_util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bed_file_name")
    parser.add_argument("window_file")
    parser.add_argument("out_file_name")
    args = parser.parse_args()
    #
    with open(args.window_file, 'r') as window_file:
        window_dicts = json.load(window_file)["windows"]
    bed = bed_util.Bed.read_bed(args.bed_file_name)
    chrom = bed.chrom
    positions_1 = bed.positions_1
    site_counts = {}
    #
    for window_id in window_dicts:
        window_dict = window_dicts[window_id]
        bounds = window_dict["bounds"]
        row_name = f"chr{chrom}_win{window_id}"
        site_counts[row_name] = np.diff(np.searchsorted(positions_1, bounds))
        print(f"SITES COUNTED IN\tWIN{window_id}\tCHR{chrom}")
    #
    header = file_util.get_header(
        chrom=chrom,
        statistic="site_counts",
        windows=window_dicts,
        bed_file=args.bed_file_name,
        cols={0: "site_counts"}
    )
    file_util.save_dict_as_arr(args.out_file_name, site_counts, header)
    print(f"SITES COUNTED ON\tCHR{chrom}")
