
#

import sys
from util import vcf_util


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    format_string = sys.argv[3]

    if sys.argv[4] != '0':
        print(sys.argv[4])
        info_string = sys.argv[4]
    else:
        info_string = None

    if sys.argv[5] == '1':
        keep_id = True
    else:
        keep_id = False

    if sys.argv[6] == '1':
        keep_filter = True
    else:
        keep_filter = False

    if sys.argv[7] == '1':
        keep_quality = True
    else:
        keep_quality = False

    vcf_util.simplify(in_path, out_path, format_string,
                      info_string=info_string, keep_id=keep_id,
                      keep_filter=keep_filter, keep_quality=keep_quality)
