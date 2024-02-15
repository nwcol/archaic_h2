
# Mostly for reading and writing my own statistics files (.json, .txt)

import json
import numpy as np
import sys
import os


# window files (JSON)






def get_header(chrom, window_id, statistic, window_dict, rows):
    header_dict = {
        "chrom": chrom,
        "window_id": window_id,
        "statistic": statistic,
        "bounds": window_dict["bounds"],
        "span": window_dict["span"],
        "n_sites": window_dict["n_sites"],
        "coverage": window_dict["coverage"],
        "right_discontinuous": window_dict["right_discontinuous"],
        "rows": rows
    }
    return str(header_dict)


def save_arr(file_name, arr, header):
    file = open(file_name, "w")
    np.savetxt(file, arr, header=header)
    file.close()
