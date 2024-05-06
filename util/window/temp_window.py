
import json
import sys


if __name__ == "__main__":

    window_file_name = sys.argv[1]
    out_file_name = sys.argv[2]

    in_file = open(window_file_name, 'r')
    in_dict = json.load(in_file)
    in_file.close()

    chrom = int(list(in_dict.keys())[0])
    in_windows = in_dict[str(chrom)]["windows"]
    window_dict = {}
    for key in in_windows:
        window_dict[int(key)] = {
            "bounds": in_windows[key]["bounds"],
            "limit_right": in_windows[key]["right_discontinuous"]
        }
    out_dict = {
        "chrom": chrom,
        "n_windows": len(in_windows),
        "windows": window_dict,
        "metadata": {}
    }
    out_file = open(out_file_name, 'w')
    json.dump(out_dict, out_file, indent=4)
    out_file.close()
