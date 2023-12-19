import sys

sys.path.insert(0, "c:/archaic/src")

from archaic import vcf_util

# WIP


def main(path, sample_id_path="c:/archaic/data/sample_ids.json"):
    header_lines = vcf_util.read_header_lines(path)
    column_titles = vcf_util.
    file = open(path)
    sample_id_map = json.load(file)
    file.close()
    for id_0 in sample_id_map:
        column_titles = column_titles.replace(id_0.encode(),
                                              sample_id_map[id_0].encode())
    return column_titles


