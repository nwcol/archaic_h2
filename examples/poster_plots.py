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
import string

from archaic import utils, plotting, simulation
from archaic.spectra import H2Spectrum


r_bins = np.logspace(-6, -2, 17)
x = r_bins[:-1] + np.diff(r_bins) / 2

_r = np.concatenate([[0], np.logspace(-7, -1, 30)])
_x = _r[:-1] + np.diff(_r) / 2


"""
util functions
"""


def make_line_legend(ax, labels, colors, columns=1):
    #
    ax.legend(
        [Line2D([0], [0], color=color, lw=2) for color in colors],
        labels,
        framealpha=0,
        fontsize='x-small',
        ncols=columns
    )


def format_H2_ax(ax):

    ax.set_ylim(0,)
    ax.set_xlim(5e-7, 2e-2)
    ax.set_xscale('log')
    ax.set_xlabel('$r$')
    ax.set_ylabel(r'$E[H_2]$')
    ax.set_xticks(np.logspace(-6, -2, 5))
    plt.minorticks_off()
    plotting.format_ticks(ax)


def label_subfigures(axs):
    # letter labels
    alphabet = string.ascii_uppercase
    for i, ax in enumerate(axs):
        ax.set_title(alphabet[i], fontsize='large', loc='left')



"""
main figure funcs
"""


def __make_figure1(fname='figures/fig1.svg'):
    # a 6-subfigure plot of expected H2 behavior

    fig_size = (7, 6)

    fig, _axs = plt.subplots(3, 2, figsize=fig_size, layout='constrained')
    axs = _axs.flat

    # steady-state and size change
    b = demes.Builder(time_units='generations')
    b.add_deme(
        'X',
        epochs=[
            dict(end_time=3e3, start_size=1e4),
            dict(end_time=0, start_size=1e4)
        ]
    )
    steady_state = b.resolve()

    b = demes.Builder(time_units='generations')
    b.add_deme(
        'X',
        epochs=[
            dict(end_time=3e3, start_size=1e4),
            dict(end_time=2e3, start_size=1e4),
            dict(end_time=0, start_size=1e4, end_size=4e4)
        ]
    )
    expansion = b.resolve()

    b = demes.Builder(time_units='generations')
    b.add_deme(
        'X',
        epochs=[
            dict(end_time=3e3, start_size=1e4),
            dict(end_time=2e3, start_size=1e4),
            dict(end_time=1e3, start_size=1e3),
            dict(end_time=0, start_size=1e4)
        ]
    )
    bottleneck = b.resolve()

    colors = ['black', 'red', 'blue']
    labels = ['steady state', 'exponential growth', 'bottleneck']

    _r = np.concatenate([[0], np.logspace(-8, -.3, 30)])
    _x = _r[:-1] + np.diff(_r) / 2
    fig_b_graphs = [steady_state, expansion, bottleneck]
    fig_b_exps = [H2Spectrum.from_demes(g, r_bins=_r).data[:-1, 0] for
                  g in fig_b_graphs]
    for i, exp in enumerate(fig_b_exps):
        demesdraw.size_history(
            fig_b_graphs[i], ax=axs[0], colours=colors[i], inf_ratio=0
        )
        axs[1].plot(
            _x,
            exp,
            color=colors[i]
        )
    axs[0].set_ylim(0, 5e4)
    axs[0].set_yticks(np.arange(0, 6e4, 1e4), [0, 1, 2, 3, 4, 5])
    axs[0].set_xlim(0, 2.5e3)
    axs[0].set_xticks(np.arange(0, 4e3, 1e3), [0, 1, 2, 3])
    make_line_legend(axs[0], labels, colors)
    axs[0].set_xlabel('$t (N_e)$')
    axs[0].set_ylabel('$N_e(t)/N_e$', rotation=90)
    axs[0].patch.set(lw=0.75, ec='black')

    axs[1].set_ylim(0,)
    axs[1].set_xscale('log')
    axs[1].set_xlabel('$r$')
    axs[1].set_ylabel(r'$E[H_2]$')
    plotting.format_ticks(axs[1])
    format_H2_ax(axs[1])


    # pulsed introgression
    Ne = 1e4
    times = [1e-4, 0.05, 0.1]
    fig_d_graphs = []
    fig_d_exps = []
    for t in times:
        _t = t * 2 * Ne
        b = demes.Builder(time_units='generations')
        b.add_deme('Ancestral', epochs=[dict(end_time=2 * Ne + _t, start_size=Ne)])
        b.add_deme(
            'X',
            ancestors=['Ancestral'],
            epochs=[dict(end_time=0, start_size=Ne)]
        )
        b.add_deme(
            'Y',
            ancestors=['Ancestral'],
            epochs=[dict(end_time=_t, start_size=Ne)]
        )
        b.add_pulse(sources=['Y'], dest='X', proportions=[0.01], time=_t)
        print(_t)
        g = b.resolve()
        fig_d_graphs.append(g)
        fig_d_exps.append(H2Spectrum.from_demes(g, r_bins=r_bins).data[:-1, 0])
    demesdraw.tubes(g, axs[2], colours='grey', labels='xticks')

    colors = ['black', 'red', 'blue', 'orange']
    # steady state
    steady_exp = H2Spectrum.from_demes(steady_state, r_bins=r_bins).data[:-1, 0]
    axs[3].plot(x, steady_exp, color=colors[0])
    for i, expectation in enumerate(fig_d_exps):
        axs[3].plot(
            x,
            expectation,
            color=colors[i + 1]
        )
    labels = ['no admixture', '$T=0$', '$T=0.05$', '$T=0.1$']
    format_H2_ax(axs[3])
    make_line_legend(axs[3], labels, colors)


    # ancestral structure
    Ne = 1e4
    times = [1e-4, 0.05, 0.1]
    fig_d_graphs = []
    fig_d_exps = []
    for t in times:
        _t = t * 2 * Ne
        b = demes.Builder(time_units='generations')
        b.add_deme('Ancestral', epochs=[dict(end_time=2 * Ne + _t, start_size=Ne)])
        b.add_deme(
            'X',
            ancestors=['Ancestral'],
            epochs=[dict(end_time=0, start_size=Ne)]
        )
        b.add_deme(
            'Y',
            ancestors=['Ancestral'],
            epochs=[dict(end_time=_t, start_size=Ne)]
        )
        b.add_pulse(sources=['Y'], dest='X', proportions=[0.01], time=_t)
        print(_t)
        g = b.resolve()
        fig_d_graphs.append(g)
        fig_d_exps.append(H2Spectrum.from_demes(g, r_bins=r_bins).data[:-1, 0])
    demesdraw.tubes(g, axs[2], colours='grey', labels='xticks')


    plt.savefig(fname, format='svg', bbox_inches='tight')
    return None


def make_figure1(fname='figures/fig1.svg'):
    # a 4-subfigure plot of expected H2 behavior
    fig_size = (7, 5)
    fig, _axs = plt.subplots(2, 2, figsize=fig_size, layout='constrained')
    axs = _axs.flat
    label_subfigures(axs)

    # steady-state and size change
    b = demes.Builder(time_units='generations')
    b.add_deme(
        'X',
        epochs=[
            dict(end_time=3e3, start_size=1e4),
            dict(end_time=0, start_size=1e4)
        ]
    )
    steady_state = b.resolve()

    b = demes.Builder(time_units='generations')
    b.add_deme(
        'X',
        epochs=[
            dict(end_time=3e3, start_size=1e4),
            dict(end_time=2e3, start_size=1e4),
            dict(end_time=0, start_size=1e4, end_size=4e4)
        ]
    )
    expansion = b.resolve()

    b = demes.Builder(time_units='generations')
    b.add_deme(
        'X',
        epochs=[
            dict(end_time=3e3, start_size=1e4),
            dict(end_time=2e3, start_size=1e4),
            dict(end_time=1e3, start_size=1e3),
            dict(end_time=0, start_size=1e4)
        ]
    )
    bottleneck = b.resolve()

    colors = ['black', 'red', 'blue']
    labels = ['steady state', 'exponential growth', 'bottleneck']
    fig_b_graphs = [steady_state, expansion, bottleneck]
    fig_b_exps = [H2Spectrum.from_demes(g, r_bins=_r).data for
                  g in fig_b_graphs]
    for i, exp in enumerate(fig_b_exps):
        demesdraw.size_history(
            fig_b_graphs[i], ax=axs[0], colours=colors[i], inf_ratio=0
        )
        axs[1].plot(
            _x,
            exp[:-1, 0],
            color=colors[i]
        )
        H = exp[-1, 0]
        axs[1].scatter([2e-8, 0.2], [2 * H **2, H **2], color=colors[i], marker='+')


    axs[0].set_ylim(0, 5e4)
    axs[0].set_yticks(np.arange(0, 6e4, 1e4), [0, 1, 2, 3, 4, 5])
    axs[0].set_xlim(0, 2.5e3)
    axs[0].set_xticks(np.arange(0, 4e3, 1e3), [0, 1, 2, 3])
    make_line_legend(axs[0], labels, colors)
    axs[0].set_xlabel('$t$ ($2N_e$)')
    axs[0].set_ylabel('$N_e(t)/N_e$', rotation=90)
    axs[0].patch.set(lw=0.75, ec='black')

    axs[1].set_ylim(0,)
    axs[1].set_xlabel('$r$')
    axs[1].set_ylabel(r'$E[H_2]$')
    plotting.format_ticks(axs[1])
    axs[1].set_xlim(5e-9, 1)
    axs[1].set_xscale('log')


    # pulsed introgression
    Ne = 1e4
    Ne2 = 2 * Ne
    gammas = [0.02, 0.02, 0.2]
    join_times = [0.05, 0.5, 0.05]
    struc_graphs = []
    struc_exps = []
    for t, gamma in zip(join_times, gammas):
        _t = t * Ne2
        b = demes.Builder(time_units='generations')
        b.add_deme('Ancestral', epochs=[dict(end_time=Ne2 + _t, start_size=Ne)])
        b.add_deme(
            'X',
            ancestors=['Ancestral'],
            epochs=[dict(end_time=0, start_size=Ne)]
        )
        b.add_deme(
            'Y',
            ancestors=['Ancestral'],
            epochs=[dict(end_time=_t, start_size=Ne)]
        )
        b.add_pulse(sources=['Y'], dest='X', proportions=[gamma], time=_t)
        print(_t)
        g = b.resolve()
        struc_graphs.append(g)
        struc_exps.append(H2Spectrum.from_demes(g, r_bins=_r).data)
    demesdraw.tubes(struc_graphs[1], axs[2], colours='grey', labels='xticks')
    axs[2].set_yticks([0, join_times[1] * Ne2, Ne2 * (1 + join_times[1])], [0, '$T$', '$1 + T$'])
    axs[2].set_ylabel('$t$ ($2N_e$)')
    colors = ['black', 'red', 'blue', 'orange', 'green']
    # steady state
    steady_exp = H2Spectrum.from_demes(steady_state, r_bins=_r).data
    axs[3].plot(_x, steady_exp[:-1, 0], color=colors[0])
    H = steady_exp[-1, 0]
    axs[3].scatter([2e-8, 0.2], [2 * H ** 2, H ** 2], color=colors[0],
                   marker='+')
    for i, expectation in enumerate(struc_exps):
        axs[3].plot(
            _x,
            expectation[:-1, 0],
            color=colors[i + 1]
        )
        H = expectation[-1, 0]
        axs[3].scatter([2e-8, 0.2], [2 * H **2, H **2], color=colors[i + 1], marker='+')
    labels = ['no admixture'] + [f'$T={t}, \gamma={g}$' for t, g in zip(join_times, gammas)]
    #format_H2_ax(axs[3])

    axs[3].set_ylim(0,)
    axs[3].set_xlabel('$r$')
    axs[3].set_ylabel(r'$E[H_2]$')
    plotting.format_ticks(axs[3])
    axs[3].set_xlim(5e-9, 1)
    axs[3].set_xscale('log')
    make_line_legend(axs[3], labels, colors, columns=2)




    plt.savefig(fname, format='svg', bbox_inches='tight')
    return None


def make_figure3(fname='figures/figure3.svg'):
    # for a structured-ancestry model, compares coalescence rates
    fig_size = (7, 5)
    fig, _axs = plt.subplots(2, 2, figsize=fig_size, layout='constrained')
    axs = _axs.flat
    label_subfigures(axs)

    # graphs
    NeAnc = 1e4
    NeX = 1e4
    NeY = 2e3
    Tjoin = 2 * NeAnc * 0.2
    Tsplit = 2 * NeAnc + Tjoin
    gamma = 0.5

    b = demes.Builder(time_units='generations')
    b.add_deme('X', epochs=[dict(end_time=0, start_size=NeX)])
    b.add_deme(
        'Y', start_time=Tsplit, ancestors=['X'],
        epochs=[dict(end_time=Tjoin, start_size=NeY)]
    )
    b.add_pulse(sources=['Y'], dest='X', proportions=[gamma], time=Tjoin)
    struc_graph = b.resolve()

    t_range = np.linspace(0, Tsplit * 1.2, 200)
    struc_rate = simulation.get_coalescent_rate(struc_graph, 'X', t_range)
    inv_rate = 1 / (2 * struc_rate)

    _t = np.array([Tjoin + 1, Tsplit - 1])
    _Nx = 1 / (2 * simulation.get_coalescent_rate(struc_graph, 'X', _t))
    b = demes.Builder(time_units='generations')
    b.add_deme(
        'X',
        epochs=[
            dict(end_time=Tsplit, start_size=NeX),
            dict(end_time=Tjoin, start_size=_Nx[1], end_size=_Nx[0]),
            dict(end_time=0, start_size=NeX)
        ]
    )
    size_graph = b.resolve()
    size_rate = simulation.get_coalescent_rate(struc_graph, 'X', t_range)

    #_t_range = np.flip(t_range)
    #NeX = np.flip(inv_rate)
    #b = demes.Builder(time_units='generations')
    #epochs = [dict(end_time=_t_range[i], start_size=NeX[i]) for i in range(len(t_range))]
    #b.add_deme('X', epochs=epochs)
    #size_graph = b.resolve()
    #size_rate = simulation.get_coalescent_rate(struc_graph, 'X', t_range)


    labels = ['structured', 'unstructured']
    _deme_colors = ['grey', 'blue']
    colors = ['black', 'blue']
    styles = ['solid', 'dashed']
    Ne2 = 2 * NeAnc

    # deme plots
    for i, graph in enumerate([struc_graph, size_graph]):
        demesdraw.tubes(
            graph, axs[i], colours=_deme_colors[i], labels='xticks', fill=True,
            max_time=Tsplit * 1.2
        )
        axs[i].set_yticks([0, Tjoin, Tsplit], [0, Tjoin/Ne2, Tsplit/Ne2])
        axs[i].set_ylabel('$t$ ($2N_e$)')

    # rate plot
    for i, rate in enumerate([struc_rate, size_rate]):
        axs[2].plot(t_range, rate, color=colors[i], linestyle=styles[i])
    axs[2].set_xlabel('$t$ ($2N_e$)')
    axs[2].set_ylabel('coalescent rate')
    axs[2].set_xlim(0, Tsplit * 1.4)
    plotting.format_ticks(axs[2])
    axs[2].set_xticks([0, Tjoin, Tsplit], [0, Tjoin/Ne2, Tsplit/Ne2])
    make_line_legend(axs[2], labels, colors)

    # H2 plot
    exps = [H2Spectrum.from_demes(g, r_bins=_r).data
            for g in [struc_graph, size_graph]]
    for i, exp in enumerate(exps):
        axs[3].plot(_x, exp[:-1, 0], color=colors[i], linestyle=styles[i])
        H = exp[-1, 0]
        axs[3].scatter([2e-8, 0.2], [2 * H ** 2, H ** 2], color=colors[i],
                       marker='+')

    axs[3].set_ylim(0,)
    axs[3].set_xlabel('$r$')
    axs[3].set_ylabel(r'$E[H_2]$')
    plotting.format_ticks(axs[3])
    axs[3].set_xlim(5e-9, 1)
    axs[3].set_xscale('log')

    axs[2].legend(
        [Line2D([0], [0], color=colors[0], lw=2),
        Line2D([0], [0], color=colors[1], lw=2, linestyle='dashed')],
        labels,
        framealpha=0,
        fontsize='x-small',
        ncols=2
    )
    plt.minorticks_off()

    plt.savefig(fname, format='svg', bbox_inches='tight')
    return None


def make_figure4(g_fname='figures/fig4_graph.svg'):
    # inferred archaic history

    # needs figure title

    graph_fname = '/home/nick/Projects/archaic/models/ND/best_fit/ND_new_bestfit_noH.yaml'
    data_fname = '/home/nick/Projects/archaic/models/H2stats.npz'

    fig_size = (7, 5)
    fig, ax = plt.subplots(figsize=fig_size, layout='constrained')

    graph = demes.load(graph_fname)
    demesdraw.tubes(graph, ax, num_lines_per_migration=0, colours='grey')
    ax.set_yticks(np.arange(0, 1.1e6, 1e5))
    plotting.format_ticks(ax)
    plt.savefig(g_fname, format='svg', bbox_inches='tight')
    return None





















make_figure3()

