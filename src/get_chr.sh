#!/bin/bash

# Parse one chromosome out of a .vcf.gz using bcftools, simplify it, and apply a mask
#

source /c/anaconda3/etc/profile.d/conda.sh
conda activate archaic_conda_env


while getopts  “:f:o:c:” opt; do
    case "${opt}" in
        f) 
			file_name=${OPTARG};;
		o)
			output_name=${OPTARG};;
		c)
			chrom=${OPTARG};;
	esac
done



