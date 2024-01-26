#!/bin/bash

# assumes that:

# all 10 sample files (.vcf.gz) are present in the directory and have been simplified and indexed
# a 1000 genomes strict mask (.bed) is present in the directory
# a genetic map (.txt) is present in the directory

# arg 1: chromosome number


n=$1

strict_mask=*$n*strict_mask.bed* 
genetic_map=*genetic_map*$n*.txt*
min_size=50


alt=chr"$n"_Altai.vcf.gz
cha=chr"$n"_Chagyrskaya.vcf.gz
den=chr"$n"_Denisova.vcf.gz
vin=chr"$n"_Vindija.vcf.gz


merged_mask=chr"$n"_merged_mask.bed
map_mask=chr"$n"_map_bed.bed
archaics=chr"$n"_archaics.vcf.gz
unrenamed_merge=chr"$n"_unrenamed_merged.vcf.gz
merged=chr"$n"_merged.vcf.gz


# archaic merge
make_beds.sh $genetic_map $alt $cha $den $vin
make_map_mask.sh $genetic_map $map_mask
intersect_beds.sh $merged_mask $min_size $strict_mask $map_mask chr"$n"_Altai.bed chr"$n"_Chagyrskaya.bed chr"$n"_Denisova.bed chr"$n"_Vindija.bed


# get sites with alternate alleles only
bcftools view -o alt_"$alt" -c 1 $alt
bcftools index alt_"$alt"
bcftools view -o alt_"$cha" -c 1 $cha
bcftools index alt_"$cha"
bcftools view -o alt_"$den" -c 1 $den
bcftools index alt_"$den"
bcftools view -o alt_"$vin" -c 1 $vin
bcftools index alt_"$vin"


# this merge only has alts in it
bcftools merge -o $archaics -R $merged_mask -0 alt_"$alt" alt_"$cha" alt_"$den" alt_"$vin"
bcftools index $archaics


# modern merge
fre=chr"$n"_French-1.vcf.gz
kho=chr"$n"_Khomani_San-2.vcf.gz
han=chr"$n"_Han-1.vcf.gz
pap=chr"$n"_Papuan-2.vcf.gz
yo1=chr"$n"_Yoruba-1.vcf.gz
yo3=chr"$n"_Yoruba-3.vcf.gz

bcftools merge -o $unrenamed_merge -R $merged_mask -0 $archaics $fre $kho $han $pap $yo1 $yo3

# rename samples
bcftools reheader -o $merged -s /c/archaic/data/metadata/sample_ids.txt $unrenamed_merge
