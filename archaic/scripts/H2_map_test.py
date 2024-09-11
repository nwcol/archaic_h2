"""

"""
import argparse
import demes
import moments
import msprime
import numpy as np

from archaic import inference, masks, two_locus, util
from archaic.parsing import parse_H2, bootstrap_H2, parse_SFS
from archaic.spectra import H2Spectrum


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
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--opt_method', default='Powell')
    parser.add_argument('-v', '--verbosity', type=int, default=1)
    parser.add_argument('--n_reps', type=int, default=100)
    parser.add_argument('--return_best', type=int, default=1)
    parser.add_argument('--cluster_id', default='0')
    parser.add_argument('--process_id', default='0')
    return parser.parse_args()


data_path = 'data'
graph_path = 'graphs'


def write_mask_file(L):
    #
    regions = np.array([[0, L]], dtype=np.int64)
    mask_fname = f'{data_path}/mask{int(L / 1e6)}Mb.bed'
    chrom_num = 'chr0'
    masks.write_regions(regions, mask_fname, chrom_num)
    return mask_fname


def write_map_file(L, cM_per_Mb):
    #
    cM = (L / 1e6) * cM_per_Mb
    map_fname = f'{data_path}/map{cM_per_Mb}cMMb.txt'
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

    rates = [0.8, 1.0, 1.2]

    maps = {c: write_map_file(args.L, c) for c in rates}

    mask_fname = write_mask_file(args.L)
    windows = np.array([[0, args.L]])
    bounds = np.array([args.L])
    r_bins = np.logspace(-6, -2, 17)

    # simulate and parse statistics
    vcf_fnames = []
    for i in range(args.n_windows):
        vcf_fname = f'{data_path}/win{i}.vcf'
        vcf_fnames.append(vcf_fname)
        coalsim(
            args.graph_fname,
            vcf_fname,
            samples,
            args.L,
            r=args.r,
            u=args.u
        )

    stat_fnames = {c: f'{tag}_{c}cM_H2.npz' for c in rates}
    for c in rates:
        H2_dicts = []
        for i in range(args.n_windows):
            H2_dicts.append(
                parse_H2(
                    mask_fname,
                    vcf_fnames[i],
                    maps[c],
                    windows=windows,
                    bounds=bounds,
                    r_bins=r_bins
                )
            )
        H2_stats = bootstrap_H2(H2_dicts)
        np.savez(stat_fnames[c], **H2_stats)

    # perturb
    for i in range(args.n_reps):
        inference.perturb_graph(
            args.graph_fname,
            args.options_fname,
            f'{graph_path}/init_rep{i}.yaml'
        )

    # H2
    fnames = {c: [] for c in rates}
    for c in rates:
        data = H2Spectrum.from_bootstrap_file(stat_fnames[c], graph=graph)
        for i in range(args.n_reps):
            in_fname = f'{graph_path}/init_rep{i}.yaml'
            out_fname = f'{graph_path}/{tag}_{c}cM_rep{i}.yaml'
            inference.optimize_H2(
                in_fname,
                args.options_fname,
                data,
                max_iter=args.max_iter,
                opt_method=args.opt_method,
                u=args.u,
                verbosity=1,
                use_H=True,
                out_fname=out_fname
            )
            fnames[c].append(out_fname)

    percentile = 1 - args.return_best / args.n_reps
    print(f'returning {percentile * 100}% highest-ll graphs')
    for c in rates:
        _, best_fit = get_best_fit_graphs(fnames[c], percentile)
        print(best_fit)
        for fname in best_fit:
            base_name = fname.split('/')[-1]
            g = demes.load(fname)
            demes.dump(g, base_name)


if __name__ == '__main__':
    main()
