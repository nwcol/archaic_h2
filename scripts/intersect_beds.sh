#!/bin/bash

# args: .bed files to intersect

source /c/anaconda3/etc/profile.d/conda.sh
conda activate archaic_conda_env


out_name=$1
min_size=$2


python /c/archaic/src/pyscripts/intersect_beds.py $out_name $min_size ${@:3}
