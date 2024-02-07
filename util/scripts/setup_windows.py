
#

import json
import numpy as np
import sys
from util import bed_util
from util import sample_sets
from util import plots


def round_Mb(bp):
    return np.round(bp / 1e6, 1)


if __name__ == "__main__":
    chrom = sys.argv[1]  # str
    out = sys.argv[2]
    bed = bed_util.Bed.get_chr(chrom)
    done = False
    windows = {}
    window_id = 0
    window_summaries = []

    min_pos = bed.min_pos
    max_pos = bed.max_pos

    # output
    print(f"COVERAGE START:\t{min_pos}\t({round_Mb(min_pos)} Mb)")
    print(f"COVERAGE END:\t{max_pos}\t({round_Mb(max_pos)} Mb)")
    coverage_str = plots.get_coverage_str(bed)
    sys.stdout.write(coverage_str + "\n")
    sys.stdout.write("\n")

    while not done:

        window_start = int(eval(input(f"WINDOW {window_id} START (Mb):\t")) * 1e6)
        window_end = int(eval(input(f"WINDOW {window_id} END (Mb):\t")) * 1e6)

        # make sure the window start is less than the window end : )
        if window_start > window_end:
            print("ERROR:\tWINDOW START EXCEEDS WINDOW END")
            continue

        # check whether window_start precedes first covered site
        if window_start < min_pos:
            print("WINDOW START INCREASED TO FIRST CALLED POSITION")
            window_start = min_pos + 1

        # check whether window_end exceeds the last covered site
        if window_end > max_pos:
            done = True
            window_end = max_pos + 1

        # check to make sure the window doesn't overlap with another window
        err = False
        for win_id in windows:
            limits = windows[win_id]["limits"]
            if limits[0] <= window_start < limits[1]:
                print(f"ERROR:\tWINDOW START INTERSECTS WINDOW {win_id}")
                err = True
            if limits[0] < window_end <= limits[1]:
                print(f"ERROR:\tWINDOW END INTERSECTS WINDOW {win_id}")
                err = True
        if err:
            continue

        # enter into dict
        n_sites = bed.interval_n_positions(window_start, window_end)
        span = window_end - window_start
        coverage = np.round(n_sites / span * 100, 2)
        windows[window_id] = {
            "limits": (window_start, window_end),
            "span": span,
            "coverage": coverage
        }
        # print info
        sys.stdout.flush()
        sys.stdout.write(coverage_str + "\n")
        spacer_low = int(np.ceil(window_start / 1e6)) - 1
        spacer_high = int(np.ceil(window_end / 1e6)) - spacer_low - 2
        sys.stdout.write(
            " " * spacer_low +
            "^" +
            " " * spacer_high +
            "^\n"
        )
        window_summary = (f"WINDOW {window_id}\t"
            + f"WINDOW START {round_Mb(window_start)} Mb\t"
            + f"WINDOW END {round_Mb(window_end)} Mb\t"
            + f"CALLED SITES {n_sites} ({round_Mb(n_sites)} Mb)\t"
            + f"COVERAGE {coverage}%\n"
        )
        sys.stdout.write(window_summary)
        window_summaries.append(window_summary)
        # up-index
        window_id += 1

    sys.stdout.write("\nWINDOW SUMMARY\n")
    for summary in window_summaries:
        sys.stdout.write(summary)

    save = input("Save windows as .json?\t")
    if save == "yes":
        wrap = {chrom: windows}
        out_file = open(out, "w")
        json.dump(wrap, out_file, indent=4)
        out_file.close()
    else:
        pass
