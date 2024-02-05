
# Create a .bed file recording the region covered by a .txt genetic map file

import sys
from util import bed_util


if __name__ == "__main__":
    map_path = sys.argv[1]
    out_path = sys.argv[2]
    mask = bed_util.Bed.read_map(map_path)
    mask.write_bed(out_path)
