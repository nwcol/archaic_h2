"""
makes plots for my research poster
"""
from bokeh.palettes import (
    TolRainbow,
    Muted,
    HighContrast3,
    MediumContrast,
    Vibrant,
    Bokeh
)
import demes
import demesdraw
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import moments
import numpy as np
import string
import os

from archaic import plotting, simulation, inference
from archaic.spectra import H2Spectrum


"""
colors etc
"""


_r = np.concatenate([[0], np.logspace(-7, -1, 30)])
_x = _r[:-1] + np.diff(_r) / 2


# make the Denisova color red
_one_sample_colors = [
    TolRainbow[4][0],
    TolRainbow[4][1],
    TolRainbow[4][3],
    TolRainbow[4][2]
]
_two_sample_colors = Muted[6]

_all_colors = [
    _one_sample_colors[0],
    _two_sample_colors[0], _two_sample_colors[1], _two_sample_colors[2],
    _one_sample_colors[1],
    _two_sample_colors[3], _two_sample_colors[4],
    _one_sample_colors[2],
    _two_sample_colors[5],
    _one_sample_colors[3]
]


_font_size = 10
_title_size = 13

_xlim = (8e-7, 1.15e-2)


"""
util functions
"""


def make_line_legend(ax, labels, colors, columns=1):
    #
    ax.legend(
        [Line2D([0], [0], color=color, lw=2) for color in colors],
        labels,
        framealpha=0,
        fontsize=_font_size,
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


def make_figure2(fname='figures/figure2.svg'):
    # a 4-subfigure plot of expected H2 behavior
    fig, _axs = plt.subplots(2, 2, figsize=(6.5, 4.5), layout='constrained')
    axs = _axs.flat
    for i, x in enumerate(['A', 'B', 'C', 'D']):
        axs[i].set_title(x, fontsize=_title_size, loc='left')
    #
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
    labels = ['constant', 'exponential growth', 'bottleneck']
    fig_b_graphs = [steady_state, expansion, bottleneck]
    fig_b_exps = [H2Spectrum.from_demes(g, r_bins=_r).data for
                  g in fig_b_graphs]
    ab_colors = HighContrast3
    for i, exp in enumerate(fig_b_exps):
        demesdraw.size_history(
            fig_b_graphs[i], ax=axs[0], colours=ab_colors[i], inf_ratio=0
        )
        axs[1].plot(
            _x,
            exp[:-1, 0],
            color=ab_colors[i]
        )
        H = exp[-1, 0]
        axs[1].scatter(
            [2e-8, 0.2], [2 * H **2, H **2], color=ab_colors[i], marker='+'
        )
    axs[0].set_ylim(0, 5e4)
    axs[0].set_yticks(np.arange(0, 6e4, 1e4), [0, 1, 2, 3, 4, 5])
    axs[0].set_xlim(0, 2.5e3)
    axs[0].set_xticks(np.arange(0, 4e3, 1e3), [0, 1, 2, 3])
    make_line_legend(axs[0], labels, ab_colors)
    axs[0].set_xlabel('$t$ ($2N_e$)')
    axs[0].set_ylabel('$N_e(t)/N_0$', rotation=90)
    axs[0].patch.set(lw=0.75, ec='black')

    axs[1].set_ylim(0,)
    axs[1].set_xlabel('$r$')
    axs[1].set_ylabel(r'$E[H_2]$')
    plotting.format_ticks(axs[1])
    axs[1].set_xlim(5e-9, 1)
    axs[1].set_xscale('log')

    # pulsed introgression
    Ne = 1e4
    gammas = [0.02, 0.2]
    join_times = [0.1, 0.1]
    struc_graphs = []
    struc_exps = []
    for t, gamma in zip(join_times, gammas):
        _t = t * (2 * Ne)
        b = demes.Builder(time_units='generations')
        b.add_deme(
            'Ancestral', epochs=[dict(end_time=(2 * Ne) + _t, start_size=Ne)]
        )
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

    demesdraw.tubes(struc_graphs[1], axs[2], colours='grey', labels=None)
    axs[2].set_yticks(
        [join_times[1] * 2 * Ne, 2 * Ne * (1 + join_times[1])],
        [join_times[1], 1 + join_times[1]]
    )
    axs[2].set_ylabel('$t$ ($2N_e$)')
    axs[2].text(1.3e4, -200, '$\gamma$')
    # steady state
    d_colors = HighContrast3
    steady_exp = H2Spectrum.from_demes(steady_state, r_bins=_r).data

    labels = [f'$\gamma={g}$' for g in [0] + gammas]
    axs[3].plot(_x, steady_exp[:-1, 0], color=d_colors[0], label=labels[0])
    H = steady_exp[-1, 0]
    axs[3].scatter(
        [2e-8, 0.2], [2 * H ** 2, H ** 2], color=d_colors[0], marker='+'
    )
    for i, expectation in enumerate(struc_exps):
        axs[3].plot(
            _x,
            expectation[:-1, 0],
            color=d_colors[i + 1],
            label=labels[i + 1]
        )
        H = expectation[-1, 0]
        axs[3].scatter(
            [2e-8, 0.2], [2 * H **2, H **2], color=d_colors[i + 1], marker='+'
        )

    axs[3].set_ylim(0,)
    axs[3].set_xlabel('$r$')
    axs[3].set_ylabel(r'$E[H_2]$')
    plotting.format_ticks(axs[3])
    axs[3].set_xlim(5e-9, 1)
    axs[3].set_xscale('log')
    axs[3].legend(fontsize=_font_size, framealpha=0)

    plt.savefig(fname, format='svg', bbox_inches='tight')
    return None


def make_figure3(fname='figures/figure3.svg'):
    # plots the best-fit base archaic history
    graph_fname = '/home/nick/Projects/archaic/models/ND/best_fit/NDbest.yaml'
    data_fname = '/home/nick/Projects/archaic/models/H2stats.npz'

    fig, axs = plt.subplot_mosaic(
        'AAB;AAC', figsize=(9.5, 5.75), layout='constrained'
    )
    for x in ['A', 'B', 'C']:
        axs[x].set_title(x, fontsize=_title_size, loc='left')

    graph = demes.load(graph_fname)
    data = H2Spectrum.from_bootstrap_file(data_fname, graph=graph)
    model = H2Spectrum.from_graph(graph, data.sample_ids, data.r, 1.297e-8)
    labels = ['Alt', 'Cha', 'Den', 'Vin']
    plotting.plot_two_panel_H2(
        model,
        data,
        labels,
        _all_colors,
        axs=(axs['B'], axs['C']),
    )
    axs['B'].legend(fontsize=10, framealpha=0)
    axs['C'].legend(
        fontsize=10, framealpha=0, ncols=2, loc='center',
        bbox_to_anchor=(0.5, 0.4)
    )
    positions = dict(
        Vindija=0,
        WestNeandertal=1e4,
        Chagyrskaya=2e4,
        Neandertal=2.6e4,
        Altai=3.9e4,
        Ancestral=4.5e4,
        Denisova=6.4e4,
    )
    color_mapping = dict(
        Ancestral='grey',
        Neandertal='grey',
        WestNeandertal='grey',
        Altai=_one_sample_colors[0],
        Chagyrskaya=_one_sample_colors[1],
        Denisova=_one_sample_colors[2],
        Vindija=_one_sample_colors[3]
    )
    demesdraw.tubes(
        graph,
        axs['A'],
        colours=color_mapping,
        positions=positions,
        labels=None
    )
    plotting.format_ticks(axs['A'])
    y = np.array([5e4, 8e4, 1.2e5, 7e4]) - 5.5e4
    names = ['Vindija', 'Chagyrskaya', 'Altai', 'Denisova']
    pos = np.array([positions[d] for d in names]) - 1e3
    labels = [name[:3] for name in names]
    for i in range(4):
        axs['A'].text(pos[i], y[i], labels[i], rotation=315)
    axs['B'].minorticks_off()
    axs['C'].minorticks_off()
    plt.savefig(fname, format='svg', bbox_inches='tight')
    return None


def make_figure4(out_fname='figures/figure4.svg'):
    # plots the base archaic fit
    matplotlib.rc('font', size=_font_size)
    graph_fname = \
        '/home/nick/Projects/archaic/models/ND/NDS/fixed_S/8-6/bestfit.yaml'
    data_fname = \
        '/home/nick/Projects/archaic/models/H2stats.npz'
    fig, axs = plt.subplot_mosaic(
        'AAB;AAC;ZZD', figsize=(9.5, 8.625), layout='constrained'
    )
    axs['Z'].remove()
    for x in ['A', 'B', 'C', 'D']:
        axs[x].set_title(x, fontsize=_title_size, loc='left')

    graph = demes.load(graph_fname)
    data = H2Spectrum.from_bootstrap_file(data_fname, graph=graph)
    model = H2Spectrum.from_graph(graph, data.sample_ids, data.r, 1.312e-8)

    labels = ['Alt', 'Cha', 'Den', 'Vin']
    plotting.plot_two_panel_H2(
        model,
        data,
        labels,
        _all_colors,
        axs=(axs['B'], axs['C']),
    )
    positions = dict(
        Vindija=0,
        WestNeandertal=1e4,
        Chagyrskaya=2e4,
        Neandertal=2.6e4,
        Altai=3.9e4,
        Ancestral=4.5e4,
        Denisova=6.4e4,
        Superarchaic=1e5
    )
    color_mapping = dict(
        Ancestral='grey',
        Neandertal='grey',
        WestNeandertal='grey',
        Superarchaic='grey',
        Altai=_one_sample_colors[0],
        Chagyrskaya=_one_sample_colors[1],
        Denisova=_one_sample_colors[2],
        Vindija=_one_sample_colors[3]
    )
    demesdraw.tubes(
        graph,
        axs['A'],
        colours=color_mapping,
        labels=None,
        positions=positions,
    )
    axs['A'].set_title('A', fontsize='large', loc='left')
    plotting.format_ticks(axs['A'])
    axs['A'].set_ylabel('$t$ (years)')
    y = np.array([5e4, 8e4, 1.2e5, 7e4, 1e5]) - 1.5e5
    names = ['Vindija', 'Chagyrskaya', 'Altai', 'Denisova', 'Superarchaic']
    pos = np.array([positions[d] for d in names]) - 1e3
    labels = [name[:3] for name in names]
    for i in range(5):
        axs['A'].text(pos[i], y[i], labels[i], rotation=315)

    axs['B'].legend(fontsize=10, framealpha=0)
    axs['C'].legend(
        fontsize=10, framealpha=0, ncols=2, loc='center',
        bbox_to_anchor=(0.5, 0.4)
    )

    D = axs['D']
    ci = 1.96
    r_bins = data.r_bins
    x = r_bins[:-1] + np.diff(r_bins) / 2
    base_graph_fname = \
        '/home/nick/Projects/archaic/models/ND/best_fit/NDbest.yaml'
    base_graph = demes.load(base_graph_fname)
    base_model = H2Spectrum.from_graph(
        base_graph, data.sample_ids, data.r, 1.297e-8
    )
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
            x, data.arr[:-1, i], yerr=y_err, color=_all_colors[i], fmt=".",
            capsize=0
        )
        D.plot(
            x, base_model.arr[:-1, i], color=_all_colors[i],
            linestyle='dashed', label=f'{label}: base model'
        )
        D.plot(
            x, model.arr[:-1, i], color=_all_colors[i],
            label=f'{label}: superarchaic model'
        )
    D.set_ylim(0,)
    D.set_xlabel('$r$')
    D.set_ylabel('$H_2$')
    D.set_xscale('log')
    D.legend(fontsize=_font_size, framealpha=0, )
    D.set_xlim(_xlim)
    D.set_yticks([0, 5e-8, 1e-7, 1.5e-7])
    plotting.format_ticks(D)
    D.minorticks_off()
    axs['B'].minorticks_off()
    axs['C'].minorticks_off()
    axs['D'].minorticks_off()
    plt.savefig(out_fname, format='svg', bbox_inches='tight')


def make_figure5(fname='figures/figure5.svg'):
    # for a structured-ancestry model, compares coalescence rates
    fig, _axs = plt.subplots(2, 2, figsize=(6.5, 4.5), layout='constrained')
    axs = _axs.flat
    label_subfigures(axs)

    # graphs
    NeAnc = 1e4
    NeX = 1e4
    NeY = 2e3
    Tjoin = 2 * NeAnc * 0.1
    Tsplit = 4 * NeAnc + Tjoin
    gamma = 0.5

    b = demes.Builder(time_units='generations')
    b.add_deme('X', epochs=[dict(end_time=0, start_size=NeX)])
    b.add_deme(
        'Y', start_time=Tsplit, ancestors=['X'],
        epochs=[dict(end_time=Tjoin, start_size=NeY)]
    )
    b.add_pulse(sources=['Y'], dest='X', proportions=[gamma], time=Tjoin)
    struc_graph = b.resolve()

    t_range = np.linspace(0, Tsplit * 1.1, 300)
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
    labels = ['structured', 'non-structured']
    _deme_colors = HighContrast3[:2]
    colors = HighContrast3[:2]
    styles = ['solid', 'dashed']
    Ne2 = 2 * NeAnc

    # deme plots
    for i, graph in enumerate([struc_graph, size_graph]):
        demesdraw.tubes(
            graph, axs[i], colours=_deme_colors[i], labels=None, fill=True,
            max_time=Tsplit * 1.1
        )
        axs[i].set_yticks([Tjoin, Tsplit], [Tjoin/Ne2, Tsplit/Ne2])
        axs[i].set_ylabel('$t$ ($2N_e$)')

    axs[0].text(7.5e3, -3000, '$\gamma=0.5$')

    # rate plot
    for i, rate in enumerate([struc_rate, size_rate]):
        axs[2].plot(t_range, rate, color=colors[i], linestyle=styles[i],)
    axs[2].set_xlabel('$t$ ($2N_e$)')
    axs[2].set_ylabel('coalescent rate')
    axs[2].set_xlim(0, Tsplit * 1.1)
    plotting.format_ticks(axs[2])
    axs[2].set_xticks([Tjoin, Tsplit], [Tjoin/Ne2, Tsplit/Ne2])
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

    axs[3].set_yticks([0, 1e-6, 2e-6])
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
        fontsize=_font_size,
    )
    plt.minorticks_off()

    plt.savefig(fname, format='svg', bbox_inches='tight')
    return None


def poster_5_search():

    NeX = 1e4
    NeYs = np.arange(2000, 4000, 1000)
    Tjoin = 2 * NeX * 0.1
    Tsplit = 2 * NeX + Tjoin
    gammas = np.arange(0.2, 1, 0.1)
    discret = 40

    def get_struc_g(gamma, NeY):

        b = demes.Builder(time_units='generations')
        b.add_deme('X', epochs=[dict(end_time=0, start_size=NeX)])
        b.add_deme(
            'Y', start_time=Tsplit, ancestors=['X'],
            epochs=[dict(end_time=Tjoin, start_size=NeY)]
        )
        b.add_pulse(sources=['Y'], dest='X', proportions=[gamma], time=Tjoin)
        g = b.resolve()
        return g

    def get_nonstruc_g(struc_g):

        t = np.linspace(0, Tsplit * 1.01, discret)
        coal_rate = simulation.get_coalescent_rate(struc_g, 'X', t)
        Nt = 1 / (2 * coal_rate)
        b = demes.Builder(time_units='generations')
        _t = np.flip(t)
        _Nt = np.flip(Nt)
        b.add_deme(
            'X',
            epochs=[dict(end_time=_t[i], start_size=_Nt[i]) for i in
                    range(len(_t))]
        )
        g = b.resolve()
        return g

    r = np.logspace(-6, -2, 11)
    x = r[:-1] + np.diff(r) / 2
    resids = []
    labels = []

    for NeY in NeYs:
        for gamma in gammas:

            g1 = get_struc_g(gamma, NeY)
            g2 = get_nonstruc_g(g1)
            label = f'({NeY}, {gamma})'
            resid = H2Spectrum.from_demes(g1, r_bins=r).data[:-1, 0] - \
                    H2Spectrum.from_demes(g2, r_bins=r).data[:-1, 0]
            labels.append(label)
            resids.append(resid)
            print(NeY, gamma)

    colors = list(matplotlib.cm.gnuplot(np.linspace(0, 0.9, len(resids))))

    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')

    for i, y in enumerate(resids):
        ax.plot(x, y, color=colors[i], label=labels[i])

    return resids, labels



def make_figure6(out_fname='figures/figure6.svg'):
    # plots SFS vs H2 identifiability
    graph_fname = \
        '/home/nick/Projects/archaic/simulations/simple_admix_id/admix.yaml'
    options_fname = \
        '/home/nick/Projects/archaic/simulations/simple_admix_id/padmix.yaml'
    graph_dir = \
        '/home/nick/Projects/archaic/simulations/simple_admix_id/graphs1'
    files = [f'{graph_dir}/{x}' for x in os.listdir(graph_dir)]
    SFS_graphs = [x for x in files if '.yaml' in x and 'SFS' in x]
    H2_graphs = [x for x in files if '.yaml' in x and 'H2_' in x]
    H2H_graphs = [x for x in files if '.yaml' in x and 'H2H' in x]
    comp_graphs = [x for x in files if '.yaml' in x and 'comp' in x]

    builder = moments.Demes.Inference._get_demes_dict(graph_fname)
    options = moments.Demes.Inference._get_params_dict(options_fname)
    pnames, p0, lower, upper = \
        moments.Demes.Inference._set_up_params_and_bounds(options, builder)
    p_arrs = []
    for graphs in [SFS_graphs, H2_graphs, H2H_graphs, comp_graphs]:
        _arr = inference.get_param_arr(graphs, options_fname, permissive=True)[1]
        arr = []
        for row in _arr:
            if not np.any(row <= lower * 1.1) and not np.any(row >= upper * .9):
                arr.append(row)
        arr = np.array(arr)
        print(arr.shape)
        p_arrs.append(arr)

    graph = demes.load(graph_fname)
    fig, axs = plt.subplot_mosaic('ABC;DEF', figsize=(6.5, 4), layout='constrained')
    axs = [axs[x] for x in axs]
    label_subfigures(axs)

    # A
    A = axs[0]
    demesdraw.tubes(graph, ax=A, colours='grey', labels=None)
    plotting.format_ticks(A, x_ax=False)
    A.set_yticks([5e5, 1e5], ['$T_0$', '$T_1$'])
    A.set_ylabel('$t$')
    # annotate A
    A.text(7.5e3, 2.5e4, '$\gamma$')
    A.text(-1500, 2.5e5, '$N_X$')
    A.text(1.2e4, 2.5e5, '$N_Y$')

    # other panels
    plot_pnames = {
        '$T_0$': 'Tsplit',
        '$T_1$': 'Tjoin',
        '$\gamma$': 'gamma',
        '$N_X$': 'NX',
        '$N_Y$': 'NY'
    }
    medianprops = dict(linewidth=2, color='black')
    for i, pname in enumerate(plot_pnames):
        ax = axs[i + 1]
        j = pnames.index(plot_pnames[pname])
        truth = p0[i]
        ax.scatter(0, truth, marker='+', color='black')
        boxes = ax.boxplot(
            [p_arr[:, j] for p_arr in p_arrs],
            vert=True,
            patch_artist=True,
            widths=[0.3] * 4,
            medianprops=medianprops
        )
        for k, patch in enumerate(boxes['boxes']):
            patch.set_facecolor('white')
        ax.set_xlim(-0.5, 4.5)
        ax.set_xticks(range(5), ['truth', 'SFS', 'H2', 'H2+H', 'SFS+H2'], rotation=45)
        ax.set_ylabel(pname)
    plotting.format_ticks(axs[1], x_ax=False)
    plotting.format_ticks(axs[2], x_ax=False)

    plt.savefig(out_fname, format='svg', bbox_inches='tight')



if __name__ == '__main__':
    matplotlib.rc('font', size=_font_size)
    #make_figure5()
