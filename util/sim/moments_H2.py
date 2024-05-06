

import argparse
import demes
import moments
import numpy as np
from util import one_locus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_file_name")
    parser.add_argument("r_file_name")
    parser.add_argument("out_file_prefix")
    parser.add_argument('-u', "--mut_rate", type=float, default=1.4e-8)
    args = parser.parse_args()
    #
    graph = demes.load(args.yaml_file_name)
    r = np.loadtxt(args.r_file_name)[1:]
    sample_ids = [x.name for x in graph.demes if x.end_time == 0]
    sample_pairs = one_locus.enumerate_pairs(sample_ids)
    ld_stats = moments.LD.LDstats.from_demes(
        graph, sampled_demes=sample_ids, theta=None, r=r, u=args.mut_rate
    )
    for sample_id in sample_ids:
        H2 = ld_stats.H2(sample_id, phased=True)
        header = str({"sample_id": sample_id, "rows": "H2"})
        np.savetxt(f"{args.out_file_prefix}_{sample_id}.txt", H2,
                   header=header)
    for id_x, id_y in sample_pairs:
        header = str({"sample_id": (id_x, id_y), "rows": "H2"})
        H2 = ld_stats.H2(id_x, id_y, phased=False)
        np.savetxt(f"{args.out_file_prefix}_{id_x},{id_y}.txt", H2,
                   header=header)
