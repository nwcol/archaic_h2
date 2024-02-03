
#

import sys
from util import bed_util


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    bed = bed_util.Bed.read_vcf(in_path)
    bed.write_bed(out_path)
