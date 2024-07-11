"""

"""


import argparse
import demes
import moments
import msprime
import numpy as np
from archaic import inference
from archaic import masks
from archaic import two_locus
from archaic.parsing import parse_H2, bootstrap_H2, parse_SFS


temp_dir = 'temp'


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph_fname', required=True)
    parser.add_argument('-p', '--params_fname', required=True)
    parser.add_argument('-o', '--out_prefix', required=True)
    parser.add_argument('-n', '--n_windows', type=int, default=200)
    parser.add_argument('-L', '--L', type=float, default=5e6)
    parser.add_argument('-max', '--max_iter', nargs='*', type=int, default=[1000])
    parser.add_argument('-opt', '--opt_methods', nargs='*', default=['Powell'])
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    parser.add_argument('-s', '--samples', nargs='*', default=None)
    parser.add_argument('-u', '--u', type=float, default=1.35e-8)
    parser.add_argument('-r', '--r', type=float, default=1e-8)
    parser.add_argument('--permute_graph', type=int, default=0)
    parser.add_argument('--cluster_id', default='0')
    parser.add_argument('--process_id', default='0')
    return parser.parse_args()


def write_mask_file(L):
    #
    regions = np.array([[0, L]], dtype=np.int64)
    mask_fname = f'{temp_dir}/mask{int(L / 1e6)}Mb.bed'
    chrom_num = 'chr0'
    masks.write_regions(regions, mask_fname, chrom_num)
    return mask_fname


def write_map_file(L, r):
    #
    cM = two_locus.map_function(r * L)
    map_fname = f'{temp_dir}/map{int(L / 1e6)}Mb.txt'
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
    return 0


def get_graph_fname(prefix, statistic, cluster, process, i):
    # get a string naming an output .yaml file
    c = p = ''
    if len(cluster) > 0:
        c = f'_{cluster}'
    if len(process) > 0:
        p = f'_{process}'
    fname = f'{prefix}_{statistic}{c}{p}_iter{i}.yaml'
    return fname


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

    vcf_fnames = []
    for i in range(args.n_windows):
        vcf_fname = f'{temp_dir}/win{i}.vcf'
        vcf_fnames.append(vcf_fname)
        coalsim(
            args.graph_fname,
            vcf_fname,
            samples,
            args.L,
            r=args.r,
            u=args.u
        )
    # replace this with something more elegant later
    win_H2_fnames = []
    for i in range(args.n_windows):
        win_H2_fname = f'{temp_dir}/win{i}_H2.npz'
        win_H2_fnames.append(win_H2_fname)
        parse_H2(
            mask_fname,
            vcf_fnames[i],
            map_fname,
            windows,
            bounds,
            r_bins,
            win_H2_fname,
        )
    H2_fname = f'H2_{tag}.npz'
    SFS_fname = f'SFS_{tag}.npz'
    bootstrap_H2(win_H2_fnames, H2_fname)
    parse_SFS(
        [mask_fname] * args.n_windows,
        vcf_fnames,
        SFS_fname,
        ref_as_ancestral=True
    )
    # get the file name for the initial graph to be used for H2 and SFS opt
    if args.permute_graph:
        graph_fname_0 = get_graph_fname(
            args.out_prefix,
            args.cluster_id,
            args.process_id,
            'permuted',
            0
        )
        inference.permute_graph(
            args.graph_fname, args.params_fname, graph_fname_0
        )
    else:
        graph_fname_0 = args.graph_fname
    # inference with H2
    graph_fname = graph_fname_0
    r_bins, data = inference.read_data(H2_fname, samples)
    for i, opt_method in enumerate(args.opt_methods):
        H2_graph, opt_info = inference.optimize(
            graph_fname,
            args.params_fname,
            data,
            r_bins,
            args.max_iter[i],
            verbosity=args.verbosity,
            u=args.u,
            use_H=True,
            use_H2=True,
            opt_method=opt_method
        )
        out_fname = get_graph_fname(
            args.out_prefix,
            args.cluster_id,
            args.process_id,
            'H2',
            i + 1
        )
        H2_graph.metadata['opt_info'] = opt_info
        demes.dump(H2_graph, out_fname)
        graph_fname = out_fname
    # inference with SFS
    # opt methods are named a bit differently in moments.Demes.Inference
    graph_fname = graph_fname_0
    opt_methods = {
        'NelderMead': 'fmin',
        'BFGS': None,
        'LBFGSB': 'lbfgsb',
        'Powell': 'powell'
    }
    SFS_file = np.load(SFS_fname)
    data = moments.Spectrum(SFS_file['SFS'], pop_ids=list(SFS_file['samples']))
    uL = SFS_file['n_sites'] * args.u
    func_calls = 0
    for i, opt_method in enumerate(args.opt_methods):
        out_fname = get_graph_fname(
            args.out_prefix,
            args.cluster_id,
            args.process_id,
            'SFS',
            i + 1
        )
        log_fname = f'{temp_dir}/log{i}.txt'
        log_file = open(log_fname, 'w')
        _, __, LL = moments.Demes.Inference.optimize(
            graph_fname,
            args.params_fname,
            data,
            maxiter=args.max_iter[i],
            verbose=args.verbosity,
            uL=uL,
            log=False,
            output=out_fname,
            output_stream=log_file,
            method=opt_methods[opt_method],
            overwrite=True
        )
        log_file.close()
        lines = open(log_fname, 'r').readlines()
        func_calls = int(lines[-1].replace(',', '').split()[0]) - func_calls
        print(''.join(lines), f'fopt: {LL}\n')
        opt_info = dict(
            method=opt_methods[opt_method],
            fopt=LL,
            iters=None,
            func_calls=func_calls,
            warnflag=None
        )
        graph = demes.load(out_fname)
        graph.metadata['opt_info'] = opt_info
        demes.dump(graph, out_fname)
        graph_fname = out_fname
    return 0


if __name__ == '__main__':
    main()
