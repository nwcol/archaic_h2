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
        if Lu != Lr:
            raise ValueError('length of u does not match length of r')
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


def get_coalescent_rate(graph, sampled_deme, t, n=2):

    demography = msprime.Demography.from_demes(graph)
    debugger = demography.debug()
    rates, probs = debugger.coalescence_rate_trajectory(t, {sampled_deme: n})
    return rates
