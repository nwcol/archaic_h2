
#

import json
import numpy as np
import sys
from util import bed_util
from util import sample_sets


if __name__ == "__main__":
    chrom = sys.argv[1]  # str
    out = sys.argv[2]
    bed = bed_util.Bed.get_chr(chrom)
    max_pos = bed.max_pos
    print(f"end of chromosome coverage:\t {max_pos}")
    done = False
    windows = {}
    i = 0
    window_start = 0

    while not done:
        task = input("Enter a new window endpoint or enter 'done'\t")
        if task != "done":
            window_end = int(eval(task))
            if window_end > max_pos:
                window_end = int(max_pos) + 1
                done = True
            # extra info
            span = window_end - window_start
            n_sites = int(bed.interval_n_positions(window_start, window_end))
            coverage = np.round(100 * n_sites / span, 2)
            # enter into dict
            windows[i] = {
                "limits": (window_start, window_end),
                "span": span,
                "n_sites": n_sites,
                "coverage": coverage
            }
            # print info
            print(f"window {i}: \t{np.round(window_start / 1e6, 1)} "
                  f"\tto {np.round(window_end / 1e6, 1)} Mb"
                  f"\tspan {np.round(span / 1e6, 1)} Mb"
                  f"\t{ np.round(n_sites / 1e6, 1)} Mb covered"
                  f"\t{coverage}% coverage")
            #
            window_start = window_end
            i += 1
            if done:
                max_Mb = np.round(max_pos / 1e6, 1)
                print(f"End of chromosome coverage reached ({max_Mb} Mb)")

        elif task == "done":
            done = True

    print("Summary")
    for key in windows:
        window = windows[key]
        window_start = window["limits"][0]
        window_end = window["limits"][1]
        span = np.round(window["span"] / 1e6, 1)
        n_sites = np.round(window["n_sites"] / 1e6, 1)
        coverage = window["coverage"]
        print(f"window {key}: {np.round(window_start / 1e6, 1)} "
              f"\tto {np.round(window_end / 1e6, 1)} Mb"
              f"\tspan {np.round(span / 1e6, 1)} Mb"
              f"\t{np.round(n_sites/ 1e6, 1)} Mb covered"
              f"\t{coverage}% coverage")

    save = input("Save windows as .json?\t")
    if save == "yes":
        windows = {chrom: windows}
        out_file = open(out, "w")
        json.dump(windows, out_file, indent=4)
        out_file.close()

    else:
        pass
