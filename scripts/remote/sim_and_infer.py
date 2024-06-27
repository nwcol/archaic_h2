"""

"""

import argparse
import demes
import numpy as np
from archaic import inference
from archaic import coalescent
from archaic import masks
from archaic import two_locus
from archaic.scripts import parse_H2
from archaic.scripts import parse_sfs
from archaic.scripts import bootstrap_H2


def get_args():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_id", default="0")
    parser.add_argument("--process_id", default="0")
    parser.add_argument("-g", "--graph_fname", required=True)
    parser.add_argument("-p", "--param_fname", required=True)
    parser.add_argument("-n", "--n_windows", type=int, required=True)
    parser.add_argument("-L", "--L", type=float, required=True)
    parser.add_argument("-o", "--out_fstem", required=True)
    parser.add_argument("-m", "--max_iter", required=True, type=int)
    parser.add_argument("-s", "--sample_names", nargs='*', required=True)
    parser.add_argument("-v", "--verbosity", type=int, default=0)
    parser.add_argument("--opt_routine", default="fmin")
    parser.add_argument("--permute_graph", type=int, default=1)
    parser.add_argument("-u", "--u", type=float, default=1.35e-8)
    parser.add_argument("-r", "--r", type=float, default=1e-8)
    return parser.parse_args()


def get_out_fnames(args):
    # get the output filename 
    h2_fname = f"h2_graph_{args.cluster_id}_{args.process_id}.yaml"
    sfs_fname = f"sfs_graph_{args.cluster_id}_{args.process_id}.yaml"
    return h2_fname, sfs_fname


def make_mask(L):

    regions = np.array([[0, L]], dtype=np.int64)
    mask_fname = f"temp/mask{int(L / 1e6)}Mb.bed"
    chrom_num = "chr0"
    masks.save_mask_regions(regions, mask_fname, chrom_num)
    return mask_fname


def make_map(L, r):

    cM = two_locus.map_function(r * L)
    map_fname = f"temp/map{int(L / 1e6)}Mb.txt"
    with open(map_fname, 'w') as file:
        file.write("Position(bp)\tRate(cM/Mb)\tMap(cM)\n")
        file.write("1\t0\t0\n")
        file.write(f"{int(L)}\t0\t{cM}")
    return map_fname


def main():
    #
    args = get_args()
    mask_fname = make_mask(args.L)
    map_fname = make_map(args.L, args.r)
    windows = np.array([[0, args.L]])
    bounds = np.array([args.L])
    r_bins = np.logspace(-6, -2, 17)
    vcf_fnames = []
    npz_fnames = []
    for i in range(args.n_windows):
        vcf_fname = f"temp/win{i}.vcf"
        vcf_fnames.append(vcf_fname)
        npz_fname = f"temp/win{i}.npz"
        npz_fnames.append(npz_fname)
        coalescent.generic_coalescent(
            args.graph_fname,
            vcf_fname,
            args.sample_names,
            args.L,
            r=args.r,
            u=args.u
        )
        parse_H2.parse(
            mask_fname,
            vcf_fname,
            map_fname,
            windows,
            bounds,
            r_bins,
            npz_fname,
        )
    bootstrap_H2.bootstrap(npz_fnames, "temp/h2.npz")
    parse_sfs.parse(vcf_fnames, "temp/sfs.npz")
    #
    h2_fname, sfs_fname = get_out_fnames(args)
    if args.permute_graph:
        graph_fname = "temp/permuted_graph.yaml"
        inference.permute_graph(args.graph_fname, args.param_fname, graph_fname)
    else:
        graph_fname = args.graph_fname
    r_bins, data = inference.read_data("temp/h2.npz", args.sample_names)
    h2_graph, etc = inference.optimize(
        graph_fname,
        args.param_fname,
        data,
        r_bins,
        args.max_iter,
        verbosity=args.verbosity,
        u=args.u,
        use_H=False,
        use_H2=True,
        opt_method="fmin"
    )
    h2_graph.metadata = dict(
        fopt=- etc[0], iters=etc[1], funcalls=etc[2], flag=etc[3]
    )
    demes.dump(h2_graph, h2_fname)
    sfs_fit = inference.sfs_infer(
        "temp/sfs.npz",
        graph_fname,
        args.param_fname,
        sfs_fname,
        args.u,
        args.L * args.n_windows,
        args.sample_names,
        method="fmin",
        max_iter=1_000,
        verbosity=args.verbosity
    )
    print(sfs_fit)
    return 0


if __name__ == "__main__":
    main()
