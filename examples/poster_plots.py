"""
makes plots for my research poster
"""
import bokeh
from bokeh.palettes import TolRainbow
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
import os

from archaic import utils, plotting, simulation, inference
from archaic.spectra import H2Spectrum


_r = np.concatenate([[0], np.logspace(-7, -1, 30)])
_x = _r[:-1] + np.diff(_r) / 2


one = TolRainbow[4]
two = TolRainbow[6]

archaic_colors = [
    one[0],
    two[0], two[1], two[2],
    one[1],
    two[3], two[4],
    one[2],
    two[5],
    one[3]
]


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


def make_figure2(fname='figures/fig1.svg'):
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


def make_figure5(fname='figures/figure3.svg'):
    # for a structured-ancestry model, compares coalescence rates
    fig_size = (7, 5)
    fig, _axs = plt.subplots(2, 2, figsize=fig_size, layout='constrained')
    axs = _axs.flat
    label_subfigures(axs)

    # graphs
    NeAnc = 1e4
    NeX = 1e4
    NeY = 1e3
    Tjoin = 2 * NeAnc * 0.1
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

    t_range = np.linspace(0, Tsplit * 1.2, 300)
    struc_rate = simulation.get_coalescent_rate(struc_graph, 'X', t_range)
    inv_rate = 1 / (2 * struc_rate)

    _t = np.array([Tjoin + 1, Tsplit - 1])
    _Nx = 1 / (2 * simulation.get_coalescent_rate(struc_graph, 'X', _t))
    b = demes.Builder(time_units='generations')

    _ts = np.flip(t_range)
    _Ns = np.flip(inv_rate)
    _Ns = np.append(_Ns, NeX)
    b.add_deme(
        'X',
        epochs=[dict(end_time=_ts[i], start_size=_Ns[i]) for i in range(len(_ts))]
    )
    size_graph = b.resolve()
    size_rate = simulation.get_coalescent_rate(struc_graph, 'X', t_range)
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
    axs[2].set_xlim(0, Tsplit * 1.2)
    plotting.format_ticks(axs[2])
    axs[2].set_xticks([0, Tjoin, Tsplit], [0, Tjoin/Ne2, Tsplit/Ne2])
    make_line_legend(axs[2], labels, colors)

    # H2 plot
    markers = ['x', '+']
    exps = [H2Spectrum.from_demes(g, r_bins=_r).data
            for g in [struc_graph, size_graph]]
    for i, exp in enumerate(exps):
        axs[3].plot(_x, exp[:-1, 0], color=colors[i], linestyle=styles[i])
        H = exp[-1, 0]
        axs[3].scatter([2e-8, 0.2], [2 * H ** 2, H ** 2], color=colors[i],
                       marker=markers[i])

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


def make_figure3(g_fname='figures/fig4_graph.svg'):
    # inferred archaic history

    # needs figure title

    graph_fname = '/home/nick/Projects/archaic/models/ND/best_fit/NDbest.yaml'
    data_fname = '/home/nick/Projects/archaic/models/H2stats.npz'

    fig_size = (9, 5.5)
    fig, axs = plt.subplot_mosaic('AAB;AAC', figsize=fig_size, layout='constrained')

    graph = demes.load(graph_fname)

    data = H2Spectrum.from_bootstrap_file(data_fname, graph=graph)
    model = H2Spectrum.from_graph(graph, data.sample_ids, data.r, 1.297e-8)
    labels = ['Alt', 'Cha', 'Den', 'Vin']
    plotting.plot_two_panel_H2(
        model,
        data,
        labels,
        archaic_colors,
        axs=(axs['B'], axs['C']),
    )
    demesdraw.tubes(graph, axs['A'], colours='grey')
    axs['A'].set_yticks(np.arange(0, 1.1e6, 1e5))
    axs['A'].set_title('A', fontsize='large', loc='left')
    plotting.format_ticks(axs['A'])
    plt.savefig(g_fname, format='svg', bbox_inches='tight')
    return None


def make_figure4(out_fname='figures/figure5.svg'):

    graph_fname = '/home/nick/Projects/archaic/models/ND/NDS/fixed_ND/pulse/fits/NDSpulse_1836367_0.yaml'
    data_fname = '/home/nick/Projects/archaic/models/H2stats.npz'

    fig_size = (9, 8.25)
    fig, axs = plt.subplot_mosaic('AAB;AAC;ZZD', figsize=fig_size, layout='constrained')
    axs['Z'].remove()

    graph = demes.load(graph_fname)

    data = H2Spectrum.from_bootstrap_file(data_fname, graph=graph)
    model = H2Spectrum.from_graph(graph, data.sample_ids, data.r, 1.297e-8)

    labels = ['Alt', 'Cha', 'Den', 'Vin']
    plotting.plot_two_panel_H2(
        model,
        data,
        labels,
        archaic_colors,
        axs=(axs['B'], axs['C']),
    )
    demesdraw.tubes(graph, axs['A'], colours='grey')
    axs['A'].set_yticks(np.arange(0, 1.1e6, 1e5))
    axs['A'].set_title('A', fontsize='large', loc='left')
    plotting.format_ticks(axs['A'])
    axs['A'].set_ylabel('$t$ (years)')


    D = axs['D']
    ci = 1.96
    r_bins = data.r_bins
    x = r_bins[:-1] + np.diff(r_bins) / 2
    base_graph_fname = '/home/nick/Projects/archaic/models/ND/best_fit/NDbest.yaml'
    base_graph = demes.load(base_graph_fname)
    base_model = H2Spectrum.from_graph(base_graph, data.sample_ids, data.r, 1.297e-8)
    # curves involving Denisova
    #idx = [7, 2, 5, 8]
    idx = [7]
    for i in idx:
        var = data.covs[:-1, i, i]
        y_err = np.sqrt(var) * ci

        id_x, id_y = data.ids[i]
        if id_x == id_y:
            label = id_x[:3]
        else:
            label = f'{id_x[:3]}-{id_y[:3]}'

        D.errorbar(
            x, data.arr[:-1, i], yerr=y_err, color=archaic_colors[i], fmt=".", capsize=0
        )

        D.plot(x, base_model.arr[:-1, i], color=archaic_colors[i], linestyle='dashed')
        D.plot(x, model.arr[:-1, i], color=archaic_colors[i], label=label)

    D.set_ylim(0,)
    D.set_xlabel('$r$')
    D.set_ylabel('$H_2$')
    plotting.format_ticks(D)
    D.set_xscale('log')
    D.legend(fontsize='x-small', framealpha=0, )
    D.set_xlim(9e-7, 2e-2)
    plt.minorticks_off()

    plt.savefig(out_fname, format='svg', bbox_inches='tight')



def make_figure6(out_fname='figures/figure6.svg'):
    # plots SFS vs H2 identifiability
    graph_fname = '/home/nick/Projects/archaic/simulations/simple_admix_id/admix.yaml'
    options_fname = '/home/nick/Projects/archaic/simulations/simple_admix_id/padmix.yaml'
    graph_dir = '/home/nick/Projects/archaic/simulations/simple_admix_id/u/graphs'
    files = [f'{graph_dir}/{x}' for x in os.listdir(graph_dir)]
    SFS_graphs = [x for x in files if '.yaml' in x and 'SFS' in x]
    H2_graphs = [x for x in files if '.yaml' in x and 'H2' in x]

    builder = moments.Demes.Inference._get_demes_dict(graph_fname)
    options = moments.Demes.Inference._get_params_dict(options_fname)
    pnames, p0, _, __ = moments.Demes.Inference._set_up_params_and_bounds(options, builder)
    _, SFS_params = inference.get_param_arr(
        SFS_graphs, options_fname, permissive=True
    )
    _, H2_params = inference.get_param_arr(
        H2_graphs, options_fname, permissive=True
    )
    graph = demes.load(graph_fname)
    fig, axs = plt.subplot_mosaic('ABC;DEF', figsize=(7, 4), layout='constrained')
    axs = [axs[x] for x in axs]
    label_subfigures(axs)

    # A
    A = axs[0]
    demesdraw.tubes(graph, ax=A, colours='grey')
    plotting.format_ticks(A, x_ax=False)
    A.set_ylabel('$t$ (years)')

    colors = ['blue', 'red']
    plot_pnames = {'T0': 'Tsplit', 'T1': 'Tjoin', '$\gamma$': 'gamma', 'NX': 'NX', 'NY': 'NY'}
    for i, pname in enumerate(plot_pnames):
        ax = axs[i + 1]
        j = pnames.index(plot_pnames[pname])
        sfs = SFS_params[:, j]
        h2 = H2_params[:, j]
        truth = p0[i]
        ax.scatter(0, truth, marker='+', color='black')
        boxes = ax.boxplot(
            [p for p in [sfs, h2]],
            vert=True,
            patch_artist=True,
            widths=[0.2, 0.2],
        )
        for k, patch in enumerate(boxes['boxes']):
            patch.set_facecolor(colors[k])
        ax.set_xlim(-0.5, 2.5)
        ax.set_xticks([0, 1, 2], ['truth', 'SFS-fit', 'H2-fit'])
        ax.set_ylabel(pname)

    plt.savefig(out_fname, format='svg', bbox_inches='tight')


make_figure4()

