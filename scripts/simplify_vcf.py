
# A script for eliminating all INFO fields and any FORMAT fields not 
# specified as arguments from .vcf files

import sys

from util import vcf_util


def main(path, *args):
    """
    :param path:
    :param args:
    :return:
    """
    vcf_util.simplify(path, None, *args)
    return 0


path = str(sys.argv[1])
args = [str(x) for x in sys.argv[2:]]
main(path, *args)

