import demes

import demesdraw

import matplotlib.pyplot as plt

import matplotlib

import msprime


pos = {
    "X": 5_000,
    "Y": 70_000,
    "XY": 35_000,
    "N": 100_000,
    "D": 130_000,
    "ND": 120_000,
    "XYND": 75_000,
    "S": 170_000,
    "Ancestral": 120_000,
}


def load_yaml(plot_ka=False):
    graph = demes.load("c:/archaic/yamls/super_archaic.yaml")
    ax = demesdraw.tubes(graph, positions=pos)
    if plot_ka:
        ax_ = ax.twinx()
        max_gen = ax.get_ylim()[1]
        max_time = max_gen * 29 / 1000
        ax_.set_ylim(0, max_time)
        ax_.set_ylabel("time ago (ka)")
    return graph


def deme_to_demog(graph):
    demog = msprime.Demography.from_demes(graph)
    return demog
