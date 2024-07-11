"""
Tests for SFS parse functions
"""


import numpy as np
import moments


def test_parse_fxn():
    #
    vcf_fname = './chr22_ancestralcoverage.vcf.gz'
    popinfo_fname = './popinfo.txt'
    my_sfs_fname = './chr22_SFS.npz'
    proj = [2] * 10

    my_sfs_file = np.load(my_sfs_fname)
    my_sfs = my_sfs_file['SFS']
    pop_ids = list(my_sfs_file['samples'])

    data_dict = moments.Misc.make_data_dict_vcf(vcf_fname, popinfo_fname)
    moments_sfs = moments.Spectrum.from_data_dict(
        data_dict, pop_ids, projections=proj, polarized=False
    )

    return 0

