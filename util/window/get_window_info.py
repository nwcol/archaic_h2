
import json
import numpy as np
import os
import sys


def round_Mb(bp):
    return np.round(bp / 1e6, 1)


if __name__ == "__main__":
    window_dir = sys.argv[1]
    json_filenames = [x for x in os.listdir(window_dir) if ".json" in x]
    window_dicts = []
    for json_filename in json_filenames:
        path = f"{window_dir}/{json_filename}"
        file = open(path, 'r')
        window_dicts.append(json.load(file))
        file.close()
    window_count = 0
    spans = []
    n_sites = []
    for window_dict in window_dicts:
        chrom = list(window_dict.keys())[0]
        windows = window_dict[chrom]["windows"]
        for window_id in windows:
            window_count += 1
            spans.append(windows[window_id]["span"])
            n_sites.append(windows[window_id]["n_sites"])

    print(f"N WINDOWS:\t{window_count}")
    print(f"MEAN SPAN:\t{round_Mb(np.mean(spans))}"
          f"\tSTD {round_Mb(np.std(spans))}"
          f"\t\tMIN {round_Mb(np.min(spans))}"
          f"\t\tMAX {round_Mb(np.max(spans))}")
    print(f"MEAN N SITES:\t{round_Mb(np.mean(n_sites))}"
          f"\tSTD {round_Mb(np.std(n_sites))}"
          f"\t\tMIN {round_Mb(np.min(n_sites))}"
          f"\t\tMAX {round_Mb(np.max(n_sites))}")
