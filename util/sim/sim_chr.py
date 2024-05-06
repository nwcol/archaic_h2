
"""
For making .vcf files using chromosome-scale coalescent simulations in msprime
"""

import argparse
import demes
import msprime
import numpy as np
from util import masks
from util import maps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_file_name")
    parser.add_argument("map_file_name")
    parser.add_argument("-n", "--n_samples", type=int, default=1)
    parser.add_argument("-o", "--out_file_name", default=None)
    parser.add_argument("-c", "--contig_id", default="1")
    parser.add_argument('-u', "--mutation_rate", type=float, default=1.4e-8)
    args = parser.parse_args()
    #
    L = maps.GeneticMap.read_txt(args.map_file_name).last_position
    rate_map = msprime.RateMap.read_hapmap(args.map_file_name, sequence_length=L)
    graph = demes.load(args.yaml_file_name)
    demography = msprime.Demography.from_demes(graph)
    sample_config = {x.name: args.n_samples for x in demography.populations}
    sample_names = list(sample_config.keys())
    #
    ts = msprime.sim_ancestry(
        samples=sample_config,
        ploidy=2,
        demography=demography,
        sequence_length=L,
        recombination_rate=rate_map,
        discrete_genome=True
    )
    mts = msprime.sim_mutations(ts, rate=args.mutation_rate)
    if args.out_file_name:
        out_file_name = args.out_file_name
    else:
        out_file_name = args.yaml_file_name.replace(".yaml", ".vcf")
    with open(out_file_name, 'w') as file:
        mts.write_vcf(
            file, individual_names=sample_names, contig_id=args.contig_id
        )
    print(f"simulation written at {args.out_file_name}")
