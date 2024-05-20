#!/bin/bash


boot_archive=$1
sample_x=$2
sample_y=$3




# make pair demography
python -m util.inference.build_pair_demog $sample_x $sample_y  -m 1e-6

# run inference on it

# interpret and write results






