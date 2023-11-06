import gzip

import numpy as np

import sys

import vcf_util as vc


def main(in_filename, out_filename, *args):
    """


    :param in_filename:
    :param out_filename:
    :param args:
    :return:
    """
    out_formats = [arg.encode() for arg in args]
    out_file = open(out_filename, 'wb')
    fmt_index, sorted_formats = vc.parse_format(in_filename, out_formats)
    fmt_bytes = vc.get_format_bytes(sorted_formats)
    sample_index = vc.get_sample_index(in_filename)
    header = vc.trim_header(in_filename, out_formats)
    for line in header:
        out_file.write(line)
    with gzip.open(in_filename, 'r') as file:
        for i, line in enumerate(file):
            if b"#" not in line:
                out_file.write(
                    vc.simplify_line(line, fmt_bytes, fmt_index, sample_index)
                )
    out_file.close()
    return i


in_filename = str(sys.argv[1])
out_filename = str(sys.argv[2])
args = [str(x) for x in sys.argv[3:]]
i = main(in_filename, out_filename, *args)
print(f"{i} lines simplified. written at {out_filename}")
