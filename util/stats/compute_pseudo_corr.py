
"""
Compute the quantity H_2 / H^2 - 1 in each r bin within each chromosome
"""

import argparse
import numpy as np
from util import file_util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("site_count_file_name")
    parser.add_argument("site_pair_count_file_name")
    parser.add_argument("het_count_file_name")
    parser.add_argument("het_pair_count_file_name")
    parser.add_argument("out_file_name")
    args = parser.parse_args()
    #
    header, site_counts = file_util.load_arr(args.site_count_file_name)
    header, site_pair_counts = file_util.load_arr(
        args.site_pair_count_file_name
    )
    header, het_counts = file_util.load_arr(args.het_count_file_name)
    header, het_pair_counts = file_util.load_arr(args.het_pair_count_file_name)
    row_names = header["rows"]
    pseudo_corr = {}
    for i in range(1, 23):
        idx = np.array([j for j in row_names if f"chr{i}_" in row_names[j]])
        chr_H = het_counts[idx].sum() / site_counts[idx].sum()
        chr_H_2 = het_pair_counts[idx].sum(0) / site_pair_counts[idx].sum(0)
        pseudo_corr[f"chr{i}"] = chr_H_2 / chr_H ** 2 - 1
    chr_H_total = het_counts.sum() / site_counts.sum()
    chr_H_2_total = het_pair_counts.sum(0) / site_pair_counts.sum(0)
    pseudo_corr["total"] = chr_H_2_total / chr_H_total ** 2 - 1
    pseudo_corr, rows = file_util.dict_to_arr(pseudo_corr)
    cols = header["cols"]
    out_header = file_util.get_header(
        sample_id=header["sample_id"],
        statistic="pseudo_corr",
        rows=rows,
        cols=cols
    )
    file_util.save_arr(args.out_file_name, pseudo_corr, out_header)
