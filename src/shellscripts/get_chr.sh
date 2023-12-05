#!/bin/bash

# Parse one chromosome out of a .vcf.gz using bcftools, simplify it, and apply a mask

# arg -f file name to parse from
# arg -o output file name
# arg -c chromosome to parse
# arg -m mask to apply

source /c/anaconda3/etc/profile.d/conda.sh
conda activate archaic_conda_env


while getopts  “:f:o:c:m:” opt; do
    case "${opt}" in
        f) 
			file_name=${OPTARG};;
		o)
			output_name=${OPTARG};;
		c)
			chrom=${OPTARG};;
		m) 
			mask_file=${OPTARG};;
	esac
done

bcftools index $file_name
bcftools view -r $chrom -o raw_chr.vcf.gz $file_name 
python /c/archaic/src/pyscripts/simplify_vcf.py raw_chr.vcf.gz simplified_chr.vcf GT
rm raw_chr.vcf.gz
bgzip simplified_chr.vcf
bcftools index simplified_chr.vcf.gz
bcftools view -R $mask_file -o $output_name simplified_chr.vcf.gz
bcftools index $output_name
rm simplified_chr.vcf.gz
rm simplified_chr.vcf.gz.csi
