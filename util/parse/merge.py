
"""
merge files holding arrays of statistics from different chromosomes
"""


import argparse
import numpy as np
from util import file_util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_file_name")
    parser.add_argument("in_file_names", nargs='*')
    args = parser.parse_args()
    #
    chroms = []
    headers = {}
    arrs = {}
    for file_name in args.in_file_names:
        header, arr = file_util.load_arr(file_name)
        chrom = header["chrom"]
        chroms.append(chrom)
        headers[chrom] = header
        arrs[chrom] = arr
    chroms.sort()
    out_arr = np.vstack([arrs[chrom] for chrom in chroms])
    rows = []
    windows = []
    for chrom in chroms:
        rows += headers[chrom]["rows"]
        windows += headers[chrom]["windows"]
    ex_header = headers[chroms[0]]
    out_header = dict(
        chroms=chroms,
        statistic=ex_header["statistic"],
        windows=windows,
        cols=ex_header["cols"],
        rows=rows
    )
    for x in ["sample_id", "sample_ids"]:
        if x in ex_header:
            out_header[x] = ex_header[x]
    file_util.save_arr(args.out_file_name, out_arr, out_header)
