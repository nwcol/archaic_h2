
import sys

sys.path.append("/home/nick/Projects/archaic/src")

import archaic.bed_util as bed_util

import archaic.map_util as map_util


def main(map_path, output_name):
    genetic_map = map_util.GeneticMap.load_txt(map_path)
    map_mask = bed_util.Bed.from_genetic_map(genetic_map)
    map_mask.write_bed(output_name)
    return 0


main(sys.argv[1], sys.argv[2])
