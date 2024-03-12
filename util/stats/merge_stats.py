
# merge files holding arrays of statistics from different chromosomes


import argparse
from util import file_util


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out_file_name")
    parser.add_argument("in_file_names", nargs='*')
    args = parser.parse_args()
    #
    chroms = []
    headers = {}
    arrs = {}
    for file_name in args.in_file_names:
        header, arr = file_util.load_arr(file_name)
        chrom = header["chrom"]
        chroms.append(chrom)
        headers[chrom] = header
        arrs[chrom] = arr
    chroms.sort()
    out_dict = {}
    for chrom in chroms:
        rows = headers[chrom]["rows"]
        arr = arrs[chrom]
        for row_idx in rows:
            row_name = rows[row_idx]
            out_dict[row_name] = arr[row_idx]
    ex_header = headers[chroms[0]]
    out_header = file_util.get_header(
        chroms=chroms,
        statistic=ex_header["statistic"],
        windows={chrom: headers[chrom]["windows"] for chrom in chroms},
        vcf_files={chrom: headers[chrom]["vcf_file"] for chrom in chroms},
        bed_files={chrom: headers[chrom]["bed_file"] for chrom in chroms},
        map_files={chrom: headers[chrom]["map_file"] for chrom in chroms},
        cols=ex_header["cols"]
    )
    for x in ["sample_id", "sample_ids"]:
        if x in ex_header:
            out_header[x] = ex_header[x]
    file_util.save_dict_as_arr(args.out_file_name, out_dict, out_header)
