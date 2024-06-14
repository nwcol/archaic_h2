
import demes
import msprime
import numpy as np


def get_rate(graph, t, sample_name, n=2):

    demography = msprime.Demography.from_demes(graph)
    debugger = demography.debug()
    rates, probs = debugger.coalescence_rate_trajectory(t, {sample_name: n})
    return rates

