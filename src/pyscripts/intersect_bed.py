
import sys

sys.path.insert(0, "c:/archaic/src")

from archaic import bed_util


def main(out, min_size, *args):
    beds = [bed_util.Bed.load_bed(arg) for arg in args]
    intersected = bed_util.intersect(*beds)
    intersected = intersected.exclude(min_size)
    intersected.write_bed(out)
    return 0


main(sys.argv[1], int(sys.argv[2]), *sys.argv[3:])
