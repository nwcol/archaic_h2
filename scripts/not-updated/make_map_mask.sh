#!/bin/bash


source /c/anaconda3/etc/profile.d/conda.sh
conda activate archaic_conda_env


map_path=$1
output_name=$2


python /c/archaic/src/pyscripts/make_map_mask.py $map_path $output_name