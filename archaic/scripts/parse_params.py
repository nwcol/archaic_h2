"""

"""


import argparse
import numpy as np
from archaic import inference


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params_fname", required=True)
    parser.add_argument("-g", "--graph_fnames", nargs='*', required=True)
    return parser.parse_args()


def printout(fields):
    
    for i, x in enumerate(fields):
        if type(x) != str:
            fields[i] = np.format_float_scientific(x, precision=2)
    fields = [str(x) for x in fields]
    print(", ".join(["%- 12s" % x for x in fields]))
    

def main():
    #
    args = get_args()
    names, arr = inference.parse_graph_params(
        args.params_fname,
        args.graph_fnames,
        permissive=True
    )
    mins = np.min(arr, axis=0)
    maxs = np.max(arr, axis=0)
    means = arr.mean(0)
    meds = np.median(arr, 0)
    stds = np.std(arr, 0)
    var_coeffs = arr.std(0) / means
    printout(["NAME", "MIN", "MED", "MEAN", "MAX", 'STD', "COEFF.VAR"])
    for i, name in enumerate(names):
        printout([name, mins[i], meds[i], means[i], maxs[i], stds[i], var_coeffs[i]])
    return 0


if __name__ == "__main__":
    main()
