"""
mostly wrappers for msprime.sim_ancestry
"""
import demes
import msprime
import numpy as np

from archaic import utils


"""
msprime simulations
"""


def increment1(x):

    return [_ + 1 for _ in x]


def simulate(
    graph,
    L=1e7,
    r=1e-8,
    u=1e-8,
    sampled_demes=None,
    out_fname=None,
    contig_id=0
):
    # simulate with constant recombination rate

    if isinstance(graph, str):
        graph = demes.load(graph)

    demography = msprime.Demography.from_demes(graph)
    if sampled_demes is None:
        sampled_demes = [d.name for d in graph.demes if d.end_time == 0]
    config = {s: 1 for s in sampled_demes}

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
                individual_names=sampled_demes,
                contig_id=str(contig_id),
                position_transform=increment1
            )
        print(
            utils.get_time(),
            f'{int(mts.sequence_length)} sites simulated '
            f'on contig {contig_id} and saved at {out_fname}'
        )
    return 0


def process_sim_data():
    # parses H2 from simulated data and


    return None


def simulate_chrom(
    graph,
    out_fname,
    u_fname=None,
    r_fname=None,
    sampled_demes=None,
    contig_id=None
):
    #
    u_rates = np.load(u_fname)['rate']
    positions = np.arange(len(u_rates) + 1)
    u_map = msprime.RateMap(position=positions, rate=u_rates)

    """
    r_positions, r_rates = two_locus.read_map_file(r_fname)
    r_positions[0] = 0
    idx = np.searchsorted(r_positions, positions[-1])
    _r_positions = np.append(r_positions[:idx], positions[-1])
    _r_rates = r_rates[:idx]
    r_map = msprime.RateMap(position=_r_positions, rate=_r_rates)
    """
    r_map = msprime.RateMap.read_hapmap(r_fname, map_col=2, position_col=0)
    r_map = r_map.slice(right=positions[-1], trim=True)

    if isinstance(graph, str):
        graph = demes.load(graph)
    demography = msprime.Demography.from_demes(graph)
    if sampled_demes is None:
        sampled_demes = [d.name for d in graph.demes if d.end_time == 0]
    config = {s: 1 for s in sampled_demes}
    print(utils.get_time(), 'simulating ancestry')
    ts = msprime.sim_ancestry(
        samples=config,
        ploidy=2,
        demography=demography,
        recombination_rate=r_map,
        discrete_genome=True,
        record_provenance=False
    )
    print(utils.get_time(), 'simulating mutation')
    mts = msprime.sim_mutations(ts, rate=u_map)

    with open(out_fname, 'w') as file:
        mts.write_vcf(
            file,
            individual_names=sampled_demes,
            contig_id=str(contig_id),
            position_transform=increment1
        )
    print(
        utils.get_time(),
        f'{int(mts.sequence_length)} sites simulated '
        f'on contig {contig_id} and saved at {out_fname}'
    )
    return 0


"""
Coalescent rates
"""


def get_coalescent_rate(graph, sampled_deme, t, n=2):

    demography = msprime.Demography.from_demes(graph)
    debugger = demography.debug()
    rates, probs = debugger.coalescence_rate_trajectory(t, {sampled_deme: n})
    return rates
