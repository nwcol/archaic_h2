#!/bin/bash

# Take simplified .vcf.gz files and merge them, excluding sites which are not
# covered in every sample. Then mask using a given mask file and exclude
# regions shorter than a given length

# arg 1: output .vfc.gz file name
# arg 2: .bed mask file to apply
# arg 3: .txt map file to apply
# arg 4: limit: the minimum continuous sequence length to include
# additional args: .vcf.gz files to merge

source /c/anaconda3/etc/profile.d/conda.sh
conda activate archaic_conda_env

out_name=$1
out_stem="${out_name:0:-7}" 
mask_file=$2
map_file=$3
length_limit=$4


i=0
for arg in "$@";
do
	if [ "$i" -ge 4 ]
	then
		if ! test -f "$arg.csi";
		then
			bcftools index "$arg"
		fi
	fi
	let i++
done


bcftools merge -o merge_0.vcf.gz -R "$mask_file" "${@:5}"
bcftools index merge_0.vcf.gz

bcftools view -o merge_1.vcf.gz -e GT=\"./.\" merge_0.vcf.gz
bcftools index merge_1.vcf.gz
rm merge_0.vcf.gz
rm merge_0.vcf.gz.csi

python /c/archaic/src/pyscripts/make_bed.py merge_1.vcf.gz "$map_file" "$out_stem.bed" "$length_limit"
bcftools view -o "$out_stem.vcf.gz" -R "$out_stem.bed" merge_1.vcf.gz
rm merge_1.vcf.gz
rm merge_1.vcf.gz.csi
