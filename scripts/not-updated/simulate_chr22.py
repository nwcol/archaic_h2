
# 

import demes

import msprime

import sys


# Constants
u = 1.5e-8
L = 5.2e7
contig = 22


# Variables
sample_config = sys.argv[1]
n_iterations = int(sys.argv[2])
map_file = sys.argv[3]
demog_file = sys.argv[4]
output_path = sys.argv[5]


rate_map = msprime.RateMap.read_hapmap(map_file, sequence_length=L)
graph = demes.load(demog_file)
demog = msprime.Demography.from_demes(graph)
demog_name = demog_file.strip(".yaml")


for i in range(n_iterations):
    ts = msprime.sim_ancestry(samples=sample_config,
                              ploidy=2,
                              demography=demog,
                              sequence_length=L,
                              recombination_rate=rate_map,
                              discrete_genome=True
                              )
    mts = msprime.sim_mutations(ts, rate=u)
    name = f"{output_path}{demog_name}_{i}.vcf"
    file = open(name, 'w')
    mts.write_vcf(file, contig_id=contig)
    file.close()
    print(f"simulation {i} complete")


print("simulations complete")

