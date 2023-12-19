#!/bin/bash

# Simplify .vcf.gz files and overwrite the originals. Keeps only the GT field

# args: .vcf.gz files to Simplify


source /c/anaconda3/etc/profile.d/conda.sh
conda activate archaic_conda_env


for arg in "$@";
do		
	stem="${arg:0:-7}" 
	python /c/archaic/src/pyscripts/simplify_vcf.py "$stem.vcf.gz" GT
	bgzip -f "$stem.vcf"
	bcftools index "$stem.vcf.gz"
done
