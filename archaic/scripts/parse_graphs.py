
import argparse
import demes
import moments.Demes.Inference as minf
import numpy as np


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
    params = minf._get_params_dict(args.params_fname)
    value_arr = []
    for fname in args.graph_fnames:
        g = minf._get_demes_dict(fname)
        print(fname)
        names, vals, _, __, = minf._set_up_params_and_bounds(params, g)
        value_arr.append(vals)
    value_arr = np.array(value_arr)
    mins = np.min(value_arr, axis=0)
    maxs = np.max(value_arr, axis=0)
    means = value_arr.mean(0)
    meds = np.median(value_arr, 0)
    varcoeffs = value_arr.std(0) / means
    printout(["NAME", "MIN", "MED", "MEAN", "MAX", "COEFF.VAR"])
    for i, name in enumerate(names):
        printout([name, mins[i], meds[i], means[i], maxs[i], varcoeffs[i]])
    return 0


if __name__ == "__main__":
    args = get_args()
    main()

