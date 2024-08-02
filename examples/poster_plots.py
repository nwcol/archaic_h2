"""
makes plots for my research poster
"""
import demes
import demesdraw
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import moments
import numpy as np
import scipy

from archaic import utils, plotting, simulation
from archaic.spectra import H2Spectrum


r_bins = np.logspace(-6, -2, 17)
x = r_bins[:-1] + np.diff(r_bins) / 2


def make_figure1(fname='figures/fig1.svg'):
    # a 6-subfigure plot of expected H2 behavior

    fig_size = (6, 7)

    fig, _axs = plt.subplots(3, 2, figsize=fig_size)  # layout='constrained')
    axs = _axs.flat

    # steady-state and size change
    b = demes.Builder(time_units='generations')
    b.add_deme('X', epochs=[dict(end_time=0, start_size=10000)])
    steady_state = b.resolve()

    b = demes.Builder(time_units='generations')
    b.add_deme(
        'X',
        epochs=[
            dict(end_time=2000, start_size=10000),
            dict(end_time=0, start_size=10000, end_size=40000)
        ]
    )
    expansion = b.resolve()

    b = demes.Builder(time_units='generations')
    b.add_deme(
        'X',
        epochs=[
            dict(end_time=2000, start_size=10000),
            dict(end_time=1000, start_size=1000),
            dict(end_time=0, start_size=10000)
        ]
    )
    bottleneck = b.resolve()

    colors = ['black', 'red', 'blue']
    labels = ['steady state', 'exponential growth', 'bottleneck']

    for i, g in enumerate([steady_state, expansion, bottleneck]):
        demesdraw.size_history(g, ax=axs[0], colours=colors[i])
        axs[1].plot(
            x,
            H2Spectrum.from_demes(g, r_bins=r_bins).data[:-1, 0],
            color=colors[i],
            label=labels[i]
        )
    axs[0].set_ylim(0, )
    axs[0].set_xlim(0, )
    axs[0].legend(framealpha=0, fontsize='small')

    axs[1].set_ylim(0,)
    axs[1].set_xlim(1e-6, 1e-2)
    axs[1].set_xscale('log')
    plotting.format_ticks(axs[1])



    # pulsed introgression



    # ancestral structure



    plt.savefig(fname, format='svg', bbox_inches='tight')
    return None


def make_figure3(fname='figures/figure3.svg'):
    #
    fig_size = (7, 7)

    fig, _axs = plt.subplots(2, 2, figsize=fig_size, layout='constrained')
    axs = _axs.flat

    plt.savefig(fname, format='svg', bbox_inches='tight')
    return None






























make_figure3()

