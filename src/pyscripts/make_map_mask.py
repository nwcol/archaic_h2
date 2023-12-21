
import sys

sys.path.insert(0, "c:/archaic/src")

from archaic import bed_util

from archaic import map_util


def main(map_path, output_name):
    genetic_map = map_util.GeneticMap.load_txt(map_path)
    map_mask = bed_util.Bed.from_genetic_map(genetic_map)
    map_mask.write_bed(output_name)
    return 0


main(sys.argv[1], sys.argv[2])
