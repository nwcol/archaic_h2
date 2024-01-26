
# Create a bed file describing continuous coverage in a .vcf or .vcf.gz file

import sys

sys.path.append("/home/nick/Projects/archaic/src")

import archaic.bed_util as bed_util

import archaic.map_util as map_util


def main(path):
    """
    Make a .bed file describing continuously covered regions in a .vcg.gz file


    If min_size, exclude regions smaller than min_size

    :param file_name:
    :param out_file_name:
    :param min_size:
    :return:
    """
    bed = bed_util.Bed.from_vcf_vectorized(path)
    bed_path = path.replace(".vcf.gz", ".bed")
    bed.write_bed(bed_path)


main(sys.argv[1])
