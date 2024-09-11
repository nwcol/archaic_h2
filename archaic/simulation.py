"""
mostly wrappers for msprime.sim_ancestry
"""
import gzip
import demes
import msprime
import numpy as np

from archaic import util


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

    print(util.get_time(), 'simulating ancestry')
    ts = msprime.sim_ancestry(
        samples=config,
        ploidy=2,
        demography=demography,
        sequence_length=int(L),
        recombination_rate=r,
        discrete_genome=True
    )
    print(util.get_time(), 'simulating mutation')
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
            util.get_time(),
            f'{int(mts.sequence_length)} sites simulated '
            f'on contig {contig_id} and saved at {out_fname}'
        )
    return 0


def simulate_chromosome(
    graph,
    out_fname,
    u=None,
    r=None,
    sampled_demes=None,
    contig_id=None,
    L=None
):
    # L 'stretches' both maps
    # u, r can be floats or .bedgraph/.txt files holding rates

    try:
        u_map = float(u)
        print(util.get_time(), f'using uniform u {u_map}')
    except:
        regions, data = util.read_bedgraph(u)
        u = data['u']
        coords = regions[:, 0]
        assert coords[0] == 0
        if coords[-1] <= L:
            edges = np.append(coords, L)
        else:
            u = u[coords < L]
            edges = np.append(coords[coords < L], L)
        u_map = msprime.RateMap(position=edges, rate=u)
        print(util.get_time(), 'loaded u-map')

    try:
        r_map = float(r)
        print(util.get_time(), f'using uniform r {r_map}')
    except:
        coords, map_vals = util.read_map_file(r)
        map_rates = np.diff(map_vals) / np.diff(coords)  # cM/bp
        map_rates /= 100
        coords[0] = 0
        if coords[-1] <= L:
            edges = np.append(coords[:-1], L)
        else:
            map_rates = map_rates[coords[:-1] < L]
            edges = np.append(coords[coords < L], L)
        r_map = msprime.RateMap(position=edges, rate=map_rates)
        print(util.get_time(), 'loaded r-map')

    if isinstance(graph, str):
        graph = demes.load(graph)

    demography = msprime.Demography.from_demes(graph)

    if sampled_demes is None:
        sampled_demes = [d.name for d in graph.demes if d.end_time == 0]

    config = {s: 1 for s in sampled_demes}

    print(util.get_time(), 'simulating ancestry')
    ts = msprime.sim_ancestry(
        samples=config,
        ploidy=2,
        demography=demography,
        recombination_rate=r_map,
        discrete_genome=True,
        record_provenance=False,
        sequence_length=L
    )
    print(util.get_time(), 'simulating mutation')
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
        util.get_time(),
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
