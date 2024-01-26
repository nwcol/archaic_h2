
import sys

sys.path.append("/home/nick/Projects/archaic/src")

import archaic.bed_util as bed_util


def main(out, min_size, *args):
    beds = [bed_util.Bed.load_bed(arg) for arg in args]
    intersected = bed_util.intersect_beds(*beds)
    intersected = intersected.exclude(min_size)
    intersected.write_bed(out)
    return 0


main(sys.argv[1], int(sys.argv[2]), *sys.argv[3:])
