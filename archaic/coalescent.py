
import demes
import msprime
import numpy as np


def get_rate(graph, t, sample_name, n=2):

    demography = msprime.Demography.from_demes(graph)
    debugger = demography.debug()
    rates, probs = debugger.coalescence_rate_trajectory(t, {sample_name: n})
    return rates


"""
Various functions for msprime coalescent simulations
"""


def generic_coalescent(graph_fname, out_fname, samples, L, n=1, r=1e-8,
                       u=1.35e-8, contig_id=0):
    # uniform r-map
    demography = msprime.Demography.from_demes(demes.load(graph_fname))
    config = {s: n for s in samples}
    ts = msprime.sim_ancestry(
        samples=config,
        ploidy=2,
        demography=demography,
        sequence_length=int(L),
        recombination_rate=r,
        discrete_genome=True
    )
    mts = msprime.sim_mutations(ts, rate=u)
    mts.write_vcf(
        out_fname, individual_names=samples, contig_id=str(contig_id)
    )
    return 0
