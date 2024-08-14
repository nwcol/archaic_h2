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
    L=None,
    r=None,
    u=None,
    sampled_demes=None,
    out_fname=None,
    contig_id=0
):
    # simulate with constant recombination rate
    # you need to give either L or one/both of r/u. r/u must match lengths...

    # load graph from file if file name was given
    if isinstance(graph, str):
        graph = demes.load(graph)

    try:
        _r = float(r)
        Lr = None
    except:
        # load rates from file
        rates = np.loadtxt(r)
        positions = np.arange(len(rates) + 1)
        Lr = len(rates)
        _r = msprime.RateMap(position=positions, rate=rates)

    try:
        _u = float(u)
        Lu = None
    except:
        if 'txt' in u:
            rates = np.loadtxt(r)
        else:
            rates = np.load(u)['rate']
        positions = np.arange(len(rates) + 1)
        Lu = len(rates)
        _u = msprime.RateMap(position=positions, rate=rates)

    if L is not None:
        if Lu is not None or Lr is not None:
            raise ValueError('you cannot provide L and u or r')
        else:
            _L = int(float(L))
    elif Lu is not None and Lr is not None:
        if Lu > Lr:
            print(
                'length of u map exceeds length of r map; '
                f'shortening to {Lr}'
            )
        elif Lu < Lr:
            print(
                'length of r map exceeds length of u map; '
                f'shortening to {Lu}'
            )
        else:
            _L = int(Lu)
    elif Lu is not None:
        _L = int(Lu)
    elif Lr is not None:
        _L = int(Lr)
    else:
        _L = None

    demography = msprime.Demography.from_demes(graph)
    if sampled_demes is None:
        sampled_demes = [d.name for d in graph.demes if d.end_time == 0]
    config = {s: 1 for s in sampled_demes}

    ts = msprime.sim_ancestry(
        samples=config,
        ploidy=2,
        demography=demography,
        sequence_length=_L,
        recombination_rate=_r,
        discrete_genome=True
    )
    mts = msprime.sim_mutations(ts, rate=_u)

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
