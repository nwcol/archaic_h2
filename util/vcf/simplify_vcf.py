
#

import argparse
import sys
from util import vcf_util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file_name")
    parser.add_argument("out_file_name")
    parser.add_argument("format_string")
    parser.add_argument("-i", "--info_string", type=str, default=None)
    parser.add_argument("-d", "--keep_id", type=bool, default=False)
    parser.add_argument("-f", "--keep_filter", type=bool, default=False)
    parser.add_argument("-q", "--keep_quality", type=bool, default=False)
    args = parser.parse_args()
    #
    vcf_util.simplify(
        args.in_file_name,
        args.out_file_name,
        args.format_string,
        info_string=args.info_string,
        keep_id=args.keep_id,
        keep_filter=args.keep_filter,
        keep_quality=args.keep_quality
    )
