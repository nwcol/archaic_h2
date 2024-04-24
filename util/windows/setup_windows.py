
#

import json
import numpy as np
import sys
import matplotlib
from util import masks
from util import sample_sets
from util import plots


def round_Mb(bp):
    return np.round(bp / 1e6, 1)


if __name__ == "__main__":
    chrom = sys.argv[1]
    out = sys.argv[2]
    bed = bed_util.Bed.read_chr(chrom)
    done = False
    flag = 0
    window_dict = {}
    window_id = 0
    last_end = 0
    window_summaries = []

    matplotlib.pyplot.show()
    bed.plot_coverage(res=5e5)

    first_pos = bed.first_position
    last_pos = bed.last_position
    n_total_sites = bed.n_positions
    n_windowed_sites = 0

    # output
    print(f"COVERAGE START:\t{first_pos}\t({round_Mb(first_pos)} Mb)")
    print(f"COVERAGE END:\t{last_pos}\t({round_Mb(last_pos)} Mb)")
    coverage_str = plots.get_coverage_str(bed)
    sys.stdout.write(coverage_str + "\n")
    sys.stdout.write("\n")

    while not done:

        entry = input(f"WINDOW {window_id} START (Mb):\t")

        if entry == "done":
            done = True
            continue

        elif entry == "x" or entry == "quit":
            done = True
            flag = 1
            continue

        window_start = int(eval(entry) * 1e6)
        window_end = int(eval(input(f"WINDOW {window_id} END (Mb):\t")) * 1e6)

        # make sure the window start is less than the window end : )
        if window_start > window_end:
            print("ERROR:\tWINDOW START EXCEEDS WINDOW END")
            continue

        # check whether window_end exceeds the last covered site
        if window_end > last_pos:
            done = True

        # check to make sure the window doesn't overlap with another window
        err = False
        for win_id in window_dict:
            bounds = window_dict[win_id]["bounds"]
            if bounds[0] <= window_start < bounds[1]:
                print(f"ERROR:\tWINDOW START INTERSECTS WINDOW {win_id}")
                err = True
            if bounds[0] < window_end <= bounds[1]:
                print(f"ERROR:\tWINDOW END INTERSECTS WINDOW {win_id}")
                err = True
        if err:
            continue

        # enter into dict
        n_sites = bed.window_site_count((window_start, window_end))
        n_windowed_sites += n_sites
        span = window_end - window_start
        coverage = np.round(n_sites / span * 100, 2)
        window_dict[window_id] = {
            "bounds": (window_start, window_end),
            "span": span,
            "n_sites": n_sites,
            "coverage": coverage,
        }
        # print info
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
            + f"START {round_Mb(window_start)} Mb\t"
            + f"END {round_Mb(window_end)} Mb\t"
            + f"SPAN {round_Mb(span)} Mb\t"
            + f"CALLED SITES {n_sites} ({round_Mb(n_sites)} Mb)\t"
            + f"COVERAGE {coverage}%\n"
        )
        sys.stdout.write(window_summary)
        sys.stdout.write("\n")
        window_summaries.append(window_summary)
        # up-index
        window_id += 1
        last_end = window_end

    n_windows = window_id

    if flag == 0:
        sys.stdout.write("\nWINDOW SUMMARY\n")
        for summary in window_summaries:
            sys.stdout.write(summary)
        n_missing_sites = int(n_total_sites - n_windowed_sites)
        sys.stdout.write(
            f"\n{n_windowed_sites} SITES OF "
            f"{n_total_sites} WINDOWED"
            f"\tMISSING {n_missing_sites} SITES\n"
        )
        # If there is a gap between windows,
        for win_id in window_dict:
            if win_id + 1 < n_windows:
                next_id = win_id + 1
                next_lower_bound = window_dict[next_id]["bounds"][0]
                if window_dict[win_id]["bounds"][1] != next_lower_bound:
                    window_dict[win_id]["right_discontinuous"] = True
                else:
                    window_dict[win_id]["right_discontinuous"] = False
            else:
                window_dict[win_id]["right_discontinuous"] = True

        save = input("Save windows as .json?\t")
        if save == "yes" or save == "":
            wrap = {
                chrom: {
                    "windows": window_dict,
                    "n_missing_sites": n_missing_sites
                }
            }
            print(out)
            out_file = open(out, "w")
            json.dump(wrap, out_file, indent=4)
            out_file.close()
        else:
            pass
    else:
        pass
