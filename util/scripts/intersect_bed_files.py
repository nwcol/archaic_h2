
#

import sys
from util import bed_util


if __name__ == "__main__":
    out_path = sys.argv[1]
    min_size = int(sys.argv[2])
    bed_file_paths = sys.argv[3].strip(' ').split(' ')
    beds = [bed_util.Bed.read_bed(path) for path in bed_file_paths]
    intersect = bed_util.intersect_beds(*beds)
    if min_size:
        intersect = intersect.exclude(min_size)
    intersect.write_bed(out_path)
