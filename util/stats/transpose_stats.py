
# for rearranging two locus stats

import argparse
import numpy as np
import sys
import os
from util import file_util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir_name")
    parser.add_argument("out_dir_name")
    parser.add_argument("-s", "--stem", default=None)
    parser.add_argument("-t", "--tag", default=None)
    args = parser.parse_args()
    out_dir_name = args.out_dir_name.rstrip("/")
    #
    dict_list = file_util.load_structured_dicts(args.in_dir_name, args.tag)
    rows, arrays = file_util.dicts_to_arr(dict_list)
    for key in arrays:
        if not args.stem:
            stem = ""
        elif args.stem == key:
            stem = ""
        else:
            stem = f"{args.stem}_"
        out_file_name = f"{out_dir_name}/{stem}{key}.txt"
        header = {"sample_id": key, "rows": rows}
        file_util.save_arr(out_file_name, arrays[key], header)
