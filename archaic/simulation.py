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

    print(utils.get_time(), 'simulating ancestry')
    ts = msprime.sim_ancestry(
        samples=config,
        ploidy=2,
        demography=demography,
        sequence_length=int(L),
        recombination_rate=r,
        discrete_genome=True
    )
    print(utils.get_time(), 'simulating mutation')
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


def set_up_u_map(positions, rates, window_size, L):
    #
    idx = np.arange(0, len(positions), window_size)
    pos_list = [0] + positions[idx].tolist() + [L]
    rate_list = [0]

    for i in range(len(idx) - 1):
        mean_rate = rates[idx[i]:idx[i + 1]].mean()
        rate_list.append(mean_rate)

    rate_list.append(rates[idx[-1]:].mean())

    assert len(pos_list) == len(rate_list) + 1
    u_map = msprime.RateMap(position=pos_list, rate=rate_list)
    return u_map


def simulate_chrom(
    graph,
    out_fname,
    u_fname=None,
    r_fname=None,
    sampled_demes=None,
    contig_id=None,
    L=None
):
    # L 'stretches' both maps

    try:
        u_map = float(u_fname)
    except:
        coeff = 1.015e-7 / 2
        file = np.load(u_fname)
        u_map = set_up_u_map(file['positions'], file['rates'] * coeff, 1000, L)

    try:
        r_map = float(r_fname)
    except:
        r_map = msprime.RateMap.read_hapmap(
            r_fname,
            map_col=2,
            position_col=0,
            sequence_length=L
        )

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
        record_provenance=False,
        sequence_length=L
    )
    print(utils.get_time(), 'simulating mutation')
    mts = msprime.sim_mutations(
        ts,
        rate=u_map,
        record_provenance=False
    )

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
