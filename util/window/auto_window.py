
import argparse
import json
import numpy as np
from util import masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bed_file_name")
    parser.add_argument("out_file_name")
    parser.add_argument("-w", "--window_size", type=int, default=1e6)
    args = parser.parse_args()
    #
    bed = bed_util.Bed.read_bed(args.bed_file_name)
    n_win = bed.last_position // args.window_size + 1
    win_size = args.window_size
    bounds = np.array(
        [
            np.arange(0, (n_win + 1) * win_size, win_size),
            np.arange(win_size, (n_win + 2) * win_size, win_size)
        ]
    ).T
    n_positions = bed.get_vec_window_position_count(bounds)
    windows = {}
    j = 0
    for i, bound in enumerate(bounds):
        window = [int(bound[0]), int(bound[1])]
        if n_positions[i] > 0:
            windows[j] = {
                "bounds": window,
                "limit_right": False
            }
            j += 1
        else:
            if j > 0:
                windows[j - 1]["limit_right"] = True
    #
    out_dict = {
        "chrom": int(bed.chrom),
        "n_windows": len(windows),
        "windows": windows,
        "metadata": {}
    }
    out_file = open(args.out_file_name, "w")
    json.dump(out_dict, out_file, indent=4)
    out_file.close()
