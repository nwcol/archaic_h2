"""

"""


import argparse
import demes
import moments
import msprime
import numpy as np
from archaic import inference, masks, two_locus, utils
from archaic.parsing import parse_H2, bootstrap_H2, parse_SFS
from archaic.spectra import H2Spectrum



data_dir = 'data'
graph_dir = 'graphs'


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph_fname', required=True)
    parser.add_argument('-p', '--params_fname', required=True)
    parser.add_argument('-o', '--out_prefix', required=True)
    parser.add_argument('-n', '--n_windows', type=int, default=100)
    parser.add_argument('-L', '--L', type=float, default=1e7)
    parser.add_argument('-s', '--samples', nargs='*', default=None)
    parser.add_argument('-u', '--u', type=float, default=1.35e-8)
    parser.add_argument('-r', '--r', type=float, default=1e-8)
    parser.add_argument('-max', '--max_iter', type=int, default=500)
    parser.add_argument('-opt', '--opt_method', default='Powell')
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    parser.add_argument('--n_reps', type=int, default=100)
    parser.add_argument('--return_best', type=int, default=1)
    parser.add_argument('--cluster_id', default='0')
    parser.add_argument('--process_id', default='0')
    return parser.parse_args()


def write_mask_file(L):
    #
    regions = np.array([[0, L]], dtype=np.int64)
    mask_fname = f'{data_dir}/mask{int(L / 1e6)}Mb.bed'
    chrom_num = 'chr0'
    masks.write_regions(regions, mask_fname, chrom_num)
    return mask_fname


def write_map_file(L, r):
    #
    cM = two_locus.map_function(r * L)
    map_fname = f'{data_dir}/map{int(L / 1e6)}Mb.txt'
    with open(map_fname, 'w') as file:
        file.write('Position(bp)\tRate(cM/Mb)\tMap(cM)\n')
        file.write('1\t0\t0\n')
        file.write(f'{int(L)}\t0\t{cM}')
    return map_fname


def coalsim(
    graph_fname,
    out_fname,
    samples,
    L,
    n=1,
    r=1e-8,
    u=1.35e-8,
    contig_id='0'
):
    # perform a coalescent simulation using msprime.sim_ancestry
    def increment1(x):
        return [_ + 1 for _ in x]
    demography = msprime.Demography.from_demes(demes.load(graph_fname))
    config = {s: n for s in samples}
    ts = msprime.sim_ancestry(
        samples=config,
        ploidy=2,
        demography=demography,
        sequence_length=L,
        recombination_rate=r,
        discrete_genome=True
    )
    mts = msprime.sim_mutations(ts, rate=u)
    with open(out_fname, 'w') as file:
        mts.write_vcf(
            file,
            individual_names=samples,
            contig_id=str(contig_id),
            position_transform=increment1
        )
    print(
        utils.get_time(),
        f'{int(L)} sites simulated and saved at {out_fname}'
    )
    return 0


def get_best_fit_graphs(graph_fnames, percentile):
    # return list of highest-LL graphs
    graphs = [demes.load(fname) for fname in graph_fnames]
    LLs = [float(graph.metadata['opt_info']['fopt']) for graph in graphs]
    threshold = np.quantile(LLs, percentile, method='linear')
    print(f'LL {percentile}th percentile: {threshold}')
    idx = np.nonzero(LLs > threshold)[0]
    for i in idx:
        print(f'{graph_fnames[i]}: LL = {LLs[i]}')
    return idx, [graph_fnames[i] for i in idx]


def H2_infer(graph_fname_0, args, data, out_fname):
    #
    graph, opt_info = inference.optimize_H2(
        graph_fname_0,
        args.params_fname,
        data,
        max_iter=args.max_iter,
        opt_method=args.opt_method,
        u=args.u,
        verbosity=args.verbosity,
        use_H=True
    )
    graph.metadata['opt_info'] = opt_info
    demes.dump(graph, out_fname)


def SFS_infer(graph_fname_0, args, data, uL, out_fname):
    #
    # opt methods are named a bit differently in moments.Demes.Inference
    opt_methods = {
        'NelderMead': 'fmin',
        'BFGS': None,
        'LBFGSB': 'lbfgsb',
        'Powell': 'powell'
    }
    _, __, LL = moments.Demes.Inference.optimize(
        graph_fname_0,
        args.params_fname,
        data,
        maxiter=args.max_iter,
        verbose=args.verbosity,
        uL=uL,
        log=False,
        output=out_fname,
        method=opt_methods[args.opt_method],
        overwrite=True
    )
    opt_info = dict(
        method=args.opt_method,
        fopt=-LL,
        iters=None,
        func_calls=None,
        warnflag=None,
        statistic='SFS'
    )
    graph = demes.load(out_fname)
    graph.metadata['opt_info'] = opt_info
    demes.dump(graph, out_fname)


def composite_infer(graph_fname_0, args, H2_data, SFS_data, L, out_fname):
    #
    if H2_data.has_H:
        H2_data = H2_data.remove_H()
    graph, opt_info = inference.optimize_super_composite(
        graph_fname_0,
        args.params_fname,
        H2_data,
        SFS_data,
        L,
        max_iter=args.max_iter,
        opt_method=args.opt_method,
        u=args.u,
        verbosity=args.verbosity
    )
    graph.metadata['opt_info'] = opt_info
    demes.dump(graph, out_fname)


def main():
    #
    args = get_args()
    graph = demes.load(args.graph_fname)
    if args.samples:
        samples = args.samples
    else:
        samples = [d.name for d in graph.demes if d.end_time == 0]
    tag = f'{args.out_prefix}_{args.cluster_id}_{args.process_id}'

    mask_fname = write_mask_file(args.L)
    map_fname = write_map_file(args.L, args.r)
    windows = np.array([[0, args.L]])
    bounds = np.array([args.L])
    r_bins = np.logspace(-6, -2, 17)

    # simulate and parse statistics
    vcf_fnames = []
    for i in range(args.n_windows):
        vcf_fname = f'{data_dir}/win{i}.vcf'
        vcf_fnames.append(vcf_fname)
        coalsim(
            args.graph_fname,
            vcf_fname,
            samples,
            args.L,
            r=args.r,
            u=args.u
        )
    H2_dicts = []
    for i in range(args.n_windows):
        H2_dicts.append(
            parse_H2(
                mask_fname,
                vcf_fnames[i],
                map_fname,
                windows=windows,
                bounds=bounds,
                r_bins=r_bins
            )
        )
    H2_data_fname = f'{tag}_H2.npz'
    H2_stats = bootstrap_H2(H2_dicts)
    np.savez(H2_data_fname, **H2_stats)
    SFS_data_fname = f'{tag}_SFS.npz'
    parse_SFS(
        [mask_fname] * args.n_windows,
        vcf_fnames,
        SFS_data_fname,
        ref_as_ancestral=True
    )

    # H2
    H2_data = H2Spectrum.from_bootstrap_file(H2_data_fname, graph=graph)
    init_fnames = []
    H2_fnames = []
    for i in range(args.n_reps):
        graph_fname_0 = f'{graph_dir}/{tag}_init_rep{i}.yaml'
        inference.permute_graph(
            args.graph_fname, args.params_fname, graph_fname_0
        )
        init_fnames.append(graph_fname_0)
        out_fname = f'{graph_dir}/{tag}_H2_rep{i}.yaml'
        H2_infer(graph_fname_0, args, H2_data, out_fname)
        H2_fnames.append(out_fname)

    # SFS
    SFS_file = np.load(SFS_data_fname)
    SFS_data = moments.Spectrum(SFS_file['SFS'], pop_ids=list(SFS_file['samples']))
    L = SFS_file['n_sites']
    uL = L * args.u
    SFS_fnames = []
    for i, graph_fname_0 in enumerate(init_fnames):
        out_fname = f'{graph_dir}/{tag}_SFS_rep{i}.yaml'
        SFS_infer(graph_fname_0, args, SFS_data, uL, out_fname)
        SFS_fnames.append(out_fname)

    # composite
    composite_fnames = []
    for i, graph_fname_0 in enumerate(init_fnames):
        out_fname = f'{graph_dir}/{tag}_composite_rep{i}.yaml'
        composite_infer(graph_fname_0, args, H2_data, SFS_data, L, out_fname)
        composite_fnames.append(out_fname)

    percentile = 1 - args.return_best / args.n_reps
    print(f'returning {percentile * 100}% highest-ll graphs')
    _, best_SFS = get_best_fit_graphs(SFS_fnames, percentile)
    print(best_SFS)
    _, best_H2 = get_best_fit_graphs(H2_fnames, percentile)
    print(best_H2)
    _, best_composite = get_best_fit_graphs(composite_fnames, percentile)
    print(best_H2)
    for fname in best_SFS + best_H2 + best_composite:
        base_name = fname.split('/')[-1]
        g = demes.load(fname)
        demes.dump(g, base_name)
    return 0


if __name__ == '__main__':
    main()
