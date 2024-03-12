
# 

import argparse
import demes
import msprime
import numpy as np
from util import bed_util
from util import map_util

"""
# For making .vcf files using chromosome-scale coalescent simulations in msprime

import demes

import msprime

import sys


def main(n_iterations, sample_config, map_path, demog_path, output_path, name,
         u=1.5e-8, seq_length=5.2e7, contig=22):

    :param n_iterations: specify the number of coalescent simulations to run
    :param sample_config: integer (one-sample case) or dict (>one-sample) 
        for samples argument in msprime.sim_ancestry()
    :param map_path: path to .txt map file
    :param demog_path: path to .yaml or .yml demography file
    :param output_path: path to output directory
    :param name: stem for output file name
    :param u: mutation rate, default 1.5e-8
    :param seq_length: sequence length (should exceed max entry in map file)
    :param contig: name for the contig in .vcf output files

    seq_length = int(seq_length)
    rate_map = msprime.RateMap.read_hapmap(map_path, sequence_length=seq_length)
    demog_graph = demes.load(demog_path)
    demog = msprime.Demography.from_demes(demog_graph)
    for i in range(n_iterations):
        ts = msprime.sim_ancestry(
                samples=sample_config,
                ploidy=2, 
                demography=demog,
                sequence_length=seq_length,
                recombination_rate=rate_map,
                discrete_genome=True
                )
        mts = msprime.sim_mutations(ts, rate=u)
        vcf_path = f"{output_path}{name}_{i}.vcf"
        file = open(vcf_path, 'w')
        mts.write_vcf(file, contig_id=contig)
        file.close()
        print(f"simulation {i + 1} / {n_iterations} complete")
    log_path = f"{output_path}{name}_log.txt"
    log_file = open(log_path, 'a')
    log_file.write("simulations complete with configuration:\n")
    log_file.write(f"n_iterations: {n_iterations}\n")
    log_file.write(f"sample_config: {sample_config}\n")
    log_file.write(f"map_path: {map_path}\n")
    log_file.write(f"demog_path: {demog_path} \n")
    log_file.write(f"demography:\n\n")
    log_file.write(demog_graph.__str__())
    log_file.write(f"\nname: {name}\n")
    log_file.write(f"u = {u}\n")
    log_file.write(f"seq_length = {seq_length}\n")
    log_file.write(f"contig: {contig}\n")
    log_file.close()
    return 0

"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path")
    parser.add_argument("map_path")
    parser.add_argument("n_samples", type=int)
    parser.add_argument('-c', "--contig_id", default="x")
    parser.add_argument('-n', "--n_iterations", default=1, type=int)
    parser.add_argument('-o', "--out_dir", default="")
    parser.add_argument('-u', "--mutation_rate", default=1.5e-8, type=float)
    #
    args = parser.parse_args()
    #
    L = map_util.GeneticMap.read_txt(args.map_path).last_position
    rate_map = msprime.RateMap.read_hapmap(args.map_path, sequence_length=L)
    graph = demes.load(args.yaml_path)
    demography = msprime.Demography.from_demes(graph)
    sample_config = {x.name: args.n_samples for x in demography.populations}
    #
    for i in range(args.n_iterations):
        ts = msprime.sim_ancestry(
            samples=sample_config,
            ploidy=2,
            demography=demography,
            sequence_length=L,
            recombination_rate=rate_map,
            discrete_genome=True
        )
        mts = msprime.sim_mutations(ts, rate=args.mutation_rate)
        file_name = f"{args.out_dir}{args.contig_id}_{i}.vcf"
        file = open(file_name, 'w')
        mts.write_vcf(file, contig_id=args.contig_id)
        file.close()
        print(f"SIMULATION {i} WRITTEN")

    print("SIMULATION COMPLETE")

