#!/bin/bash

# args: .vcf.gz files to make .bed files for

source /c/anaconda3/etc/profile.d/conda.sh
conda activate archaic_conda_env


genetic_map=$1


for arg in ${@:2};
do
	python /c/archaic/src/pyscripts/make_bed.py $arg $genetic_map
done

