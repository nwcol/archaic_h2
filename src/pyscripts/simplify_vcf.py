
# Script for eliminating superfluous data from .vcf files

import gzip

import numpy as np

import sys

sys.path.insert(0, "c:/archaic/src")

from archaic import vcf_util


def main(path, *args):
    """
    :param path:
    :param args:
    :return:
    """
    i = vcf_util.simplify(path, None, *args)
    print(i)


path = str(sys.argv[1])
args = [str(x) for x in sys.argv[2:]]
main(path, *args)
