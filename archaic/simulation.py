"""
mostly wrappers for msprime.sim_ancestry
"""
import demes
import msprime
import numpy as np

from archaic import two_locus, utils


"""
msprime simulations
"""


def increment1(x):

    return [_ + 1 for _ in x]


def simulate(
    graph,
    L,
    r=1e-8,
    u=1.35e-8,
    sample_ids=None,
    out_fname=None,
    contig=0
):
    # simulate with constant recombination rate
    if isinstance(graph, str):
        graph = demes.load(graph)
    demography = msprime.Demography.from_demes(graph)
    if sample_ids is None:
        sample_ids = [d.name for d in graph.demes if d.end_time == 0]
    config = {s: 1 for s in sample_ids}

    ts = msprime.sim_ancestry(
        samples=config,
        ploidy=2,
        demography=demography,
        sequence_length=int(L),
        recombination_rate=r,
        discrete_genome=True
    )
    mts = msprime.sim_mutations(ts, rate=u)

    if out_fname is None:
        return mts
    else:
        with open(out_fname, 'w') as file:
            mts.write_vcf(
                file,
                individual_names=sample_ids,
                contig_id=str(contig),
                position_transform=increment1
            )
        print(
            utils.get_time(),
            f'{int(mts.sequence_length)} sites simulated '
            f'on contig {contig} and saved at {out_fname}'
        )
    return 0


def simulate_chrom(
    graph,
    map_fname,
    u=1.35e-8,
    sample_ids=None,
    out_fname=None,
    contig=0
):
    # simulate on a genetic map
    # get the length of the map
    header_dict = two_locus.parse_map_file_header(map_fname)
    rate_map = msprime.RateMap.read_hapmap(
        map_fname,
        position_col=header_dict['Position(bp)'],
        map_col=header_dict['Map(cM)']
    )
    if isinstance(graph, str):
        graph = demes.load(graph)
    demography = msprime.Demography.from_demes(graph)
    if sample_ids is None:
        sample_ids = [d.name for d in graph.demes if d.end_time == 0]
    config = {s: 1 for s in sample_ids}

    ts = msprime.sim_ancestry(
        samples=config,
        ploidy=2,
        demography=demography,
        recombination_rate=rate_map,
        discrete_genome=True
    )
    mts = msprime.sim_mutations(ts, rate=u)

    if out_fname is None:
        return mts
    else:
        with open(out_fname, 'w') as file:
            mts.write_vcf(
                file,
                individual_names=sample_ids,
                contig_id=str(contig),
                position_transform=increment1
            )
        print(
            utils.get_time(),
            f'{int(mts.sequence_length)} sites simulated '
            f'on contig {contig} and saved at {out_fname}'
        )
    return 0


"""
Coalescent rates
"""


def get_coalescent_rate(graph, t, sampled_deme, n=2):

    demography = msprime.Demography.from_demes(graph)
    debugger = demography.debug()
    rates, probs = debugger.coalescence_rate_trajectory(t, {sampled_deme: n})
    return rates
