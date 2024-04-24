
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
    parser.add_argument("yaml_path")
    parser.add_argument("map_path")
    parser.add_argument("n_samples", type=int)
    parser.add_argument("out_file_name")
    parser.add_argument("-c", "--contig_id", default="1")
    parser.add_argument('-u', "--mutation_rate", default=1.4e-8, type=float)
    args = parser.parse_args()
    #
    L = maps.GeneticMap.read_txt(args.map_path).last_position
    rate_map = msprime.RateMap.read_hapmap(args.map_path, sequence_length=L)
    graph = demes.load(args.yaml_path)
    demography = msprime.Demography.from_demes(graph)
    sample_config = {x.name: args.n_samples for x in demography.populations}
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
    with open(args.out_file_name, 'w') as file:
        mts.write_vcf(file, contig_id=args.contig_id)
    print(f"simulation written at {args.out_file_name}")

