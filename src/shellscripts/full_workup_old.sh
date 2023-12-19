#!/bin/bash

# this script makes the following assumptions about the directory where it is employed:

# there are 10 .vcf.gz files named
# chr*_Altai.vcf.gz
# chr*_Chagyrskaya.vcf.gz
# chr*_Denisova.vcf.gz
# chr*_Vindija.vcf.gz
# chr*_French-1.vcf.gz
# chr*_Han-1.vcf.gz
# chr*_Khomani_San-2.vcf.gz
# chr*_Papuan-2.vcf.gz
# chr*_Yoruba-1.vcf.gz
# chr*_Yoruba-3.vcf.gz

# there is a 1000 genomes strict mask file with a name of the form
# *chr*.strict_mask.bed*

# there is a map file with a name of the form 
# genetic_map*chr*.txt


# argument 1: the chromosome number. ensures you didnt place an incorrect file in the directory


chr=$1


# simplify_chr.sh *$chr*.vcf.gz

merge_chr.sh chr"$chr"_archaics.vcf.gz *$chr*mask*.bed* *map*$chr*.txt 50 *chr$chr*Altai.vcf.gz *chr$chr*Chagyrskaya.vcf.gz *chr$chr*Denisova.vcf.gz *chr$chr*Vindija.vcf.gz

mv chr"$chr"_archaics.bed chr"$chr"_merged.bed

add_chr.sh chr"$chr"_merged.vcf.gz chr"$chr"_archaics.vcf.gz chr"$chr"_merged.bed *chr$chr*French-1.vcf.gz *chr$chr*Han-1.vcf.gz *chr$chr*Khomani_San-2.vcf.gz *chr$chr*Papuan-2.vcf.gz *chr$chr*Yoruba-1.vcf.gz *chr$chr*Yoruba-3.vcf.gz 

