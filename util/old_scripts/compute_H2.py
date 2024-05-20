
import argparse
import numpy as np
from util import file_util


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("site_pair_file")
    parser.add_argument("het_pair_files", nargs="*")
    args = parser.parse_args()
    #
    site_pairs = np.loadtxt(args.site_pair_file).sum(0)
    for file_name in args.het_pair_files:
        header, het_pairs = file_util.load_arr(file_name)
        H2 = het_pairs.sum(0) / site_pairs
        if "het_pairs" in file_name:
            out_file_name = file_name.replace("het_pairs", "H2")
        else:
            out_file_name = file_name.replace(".txt", "_H2.txt")
        np.savetxt(out_file_name, H2, header=str(header))
