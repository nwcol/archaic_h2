import numpy as np

import sys


def main(arr_path, sample_size, n_resamplings, output_path,
         conf_intervals=[0.95]):
    """
    :param sample_size: number of vectors to resample per iteration
    :param n_resamplings: number of resamplings to perform
    :param conf_intervals: confidence intervals to return
    """
    arr = np.loadtxt(arr_path)
    n_rows = np.shape(arr)[0]
    n_cols = np.shape(arr)[1]
    sample_means = np.zeros((n_resamplings, n_cols))

    for i in range(n_resamplings):
        sample_idx = np.random.choice(np.arange(n_rows), sample_size)
        mean = np.mean(arr[sample_idx], axis=0)
        sample_means[i] = mean

    sample_means.sort(axis=0)
    out_arr = [np.mean(sample_means, axis=0), 
               sample_means[int(0.5 * n_rows)],
               sample_means[-1],
               sample_means[0]]
    stats = ["mean", "median", "maximum", "minimum"]

    for c in conf_intervals:
        out_arr.append(sample_means[int(n_resamplings * c)])
        out_arr.append(sample_means[int(n_resamplings * (1 - c))])
        stats.append(f"CI{c}")
        stats.append(f"CI({1-c})")

    file = open(output_path, 'w')
    out_arr = np.array(out_arr)
    np.savetxt(file, np.array(out_arr), 
               header=f"N={sample_size},B={n_resamplings} {' '.join(stats)}")
    file.close()
    return 0


arr_path = sys.argv[1]
sample_size = int(sys.argv[2])
n_resamplings = int(sys.argv[3])
output_path = sys.argv[4]


main(arr_path, sample_size, n_resamplings, output_path)

