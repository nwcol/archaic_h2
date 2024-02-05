
# mostly for fun

import numpy as np
import sys
from util import bed_util


if __name__ == "__main__":
    bed_path = sys.argv[1]
    bed = bed_util.Bed.read_bed(bed_path)
    print(f"chromosome {bed.chrom}")
    positions = bed.get_0_idx_positions()
    approx_max = np.round(bed.max_pos, -6) + 1e6
    out = []
    for i in np.arange(0, approx_max, 1e6, dtype=np.int64):
        n_positions = np.searchsorted(positions, i + 1e6) - \
            np.searchsorted(positions, i)
        coverage = n_positions / 1e6
        if coverage < 0.01:
            out.append("~")
        elif coverage < 0.1:
            out.append("-")
        elif coverage < 0.25:
            out.append("=")
        elif coverage < 0.5:
            out.append("*")
        elif coverage < 0.75:
            out.append("x")
        else:
            out.append("X")
    out = "".join(out)
    print(out)
    print(f"covered: {bed.n_positions / 1e6} Mb")
    print(f"min position: {bed.min_pos / 1e6} Mb")
    print(f"max position: {bed.max_pos / 1e6} Mb")
