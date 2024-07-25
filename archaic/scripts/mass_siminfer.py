"""
fits models using H2, H2+H, SFS, and composite methods
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
    parser.add_argument('-p', '--options_fname', required=True)
    parser.add_argument('-o', '--out_prefix', required=True)
    parser.add_argument('-n', '--n_windows', type=int, default=100)
    parser.add_argument('-L', '--L', type=float, default=1e7)
    parser.add_argument('-s', '--samples', nargs='*', default=None)
    parser.add_argument('-u', '--u', type=float, default=1.35e-8)
    parser.add_argument('-r', '--r', type=float, default=1e-8)
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--method', default='Powell')
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    parser.add_argument('--n_reps', type=int, default=100)
    parser.add_argument('--return_best', type=int, default=1)
    parser.add_argument('--fit', nargs='*', default=['H2', 'H2H', 'SFS', 'comp'])
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
    cM = two_locus.map_function(r) * L
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

    # perturb the initial graph to get starting points
    inits = []
    for i in range(args.n_reps):
        fname = f'{graph_dir}/{tag}_init_rep{i}.yaml'
        inference.perturb_graph(
            args.graph_fname, args.options_fname, out_fname=fname
        )
        inits.append(fname)

    # load statistics
    H2_data = H2Spectrum.from_bootstrap_file(H2_data_fname, graph=graph)
    SFS_data, L = inference.read_SFS(SFS_data_fname, H2_data.sample_ids)

    fits = dict(H2=[], H2H=[], SFS=[], comp=[])
    for i in range(args.n_reps):
        H2_fname = f'{graph_dir}/{tag}_H2_rep{i}.yaml'
        inference.fit_H2(
            inits[i],
            args.options_fname,
            H2_data,
            max_iter=args.max_iter,
            method=args.method,
            u=args.u,
            verbosity=args.verbosity,
            use_H=False,
            out_fname=H2_fname
        )
        fits['H2'].append(H2_fname)

        H2H_fname = f'{graph_dir}/{tag}_H2H_rep{i}.yaml'
        inference.fit_H2(
            inits[i],
            args.options_fname,
            H2_data,
            max_iter=args.max_iter,
            method=args.method,
            u=args.u,
            verbosity=args.verbosity,
            use_H=True,
            out_fname=H2H_fname
        )
        fits['H2H'].append(H2H_fname)

        SFS_fname = f'{graph_dir}/{tag}_SFS_rep{i}.yaml'
        inference.fit_SFS(
            inits[i],
            args.options_fname,
            SFS_data,
            args.u * L,
            max_iter=args.max_iter,
            method=args.method,
            verbosity=args.verbosity,
            out_fname=SFS_fname
        )
        fits['SFS'].append(SFS_fname)
        
        comp_fname = f'{graph_dir}/{tag}_comp_rep{i}.yaml'
        inference.fit_composite(
            inits[i],
            args.options_fname,
            H2_data,
            SFS_data,
            L,
            max_iter=args.max_iter,
            method=args.method,
            u=args.u,
            verbosity=args.verbosity,
            out_fname=comp_fname
        )
        fits['comp'].append(comp_fname)

    percentile = 1 - args.return_best / args.n_reps
    print(f'returning {percentile * 100}% highest-ll graphs')

    for x in fits:
        _, best_fits = get_best_fit_graphs(fits[x], percentile)
        print(x, best_fits)
        for fname in best_fits:
            base_name = fname.split('/')[-1]
            g = demes.load(fname)
            demes.dump(g, base_name)
    return 0


if __name__ == '__main__':
    main()
