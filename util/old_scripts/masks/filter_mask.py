
"""
Copy the lines for a single chromosome out of a multi-chromosome mask file
"""

import argparse
import gzip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file_name")
    parser.add_argument("out_file_name")
    parser.add_argument("chrom")
    args = parser.parse_args()
    #
    if ".gz" in args.in_file_name:
        chrom_bytes = b'chr' + args.chrom.encode()
        with open(args.out_file_name, 'wb') as out_file:
            with gzip.open(args.in_file_name, "r") as in_file:
                for line in in_file:
                    if chrom_bytes in line:
                        out_file.write(line)



