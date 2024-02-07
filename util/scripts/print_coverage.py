
# mostly for fun

import numpy as np
import sys
from util import bed_util
from util import plots


if __name__ == "__main__":
    chrom = sys.argv[1]
    bed = bed_util.Bed.get_chr(chrom)
    print(f"chromosome {bed.chrom}")
    print(plots.get_coverage_str(bed))
    print(f"covered: {np.round(bed.n_positions / 1e6, 1)} Mb")
    print(f"min position: {np.round(bed.min_pos / 1e6, 1)} Mb")
    print(f"max position: {np.round(bed.max_pos / 1e6, 1)} Mb")
