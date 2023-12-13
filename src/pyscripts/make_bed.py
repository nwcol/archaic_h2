
# Create a bed file describing continuous coverage in a .vcf or .vcf.gz file

import gzip

import sys

sys.path.insert(0, "c:/archaic/src")

from archaic import vcf_util

from archaic import bed_util


def main(path, out, min_size=None):
    """
    Make a .bed file describing continuously covered regions in a .vcg.gz file


    If min_size, exclude regions smaller than min_size

    :param file_name:
    :param out_file_name:
    :param min_size:
    :return:
    """
    bed = bed_util.Bed.from_vcf(path)
    if min_size:
        bed = bed.exclude(min_size)
    bed.write_bed(out)


path = str(sys.argv[1])
out = str(sys.argv[2])
if len(sys.argv) == 4:
    min_size = int(sys.argv[3])
else:
    min_size = None
main(path, out, min_size=min_size)
