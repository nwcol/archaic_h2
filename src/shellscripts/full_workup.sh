# assumes that:

# all 10 sample files (.vcf.gz) are present in the directory and have been simplified and indexed
# a 1000 genomes strict mask (.bed) is present in the directory
# a genetic map (.txt) is present in the directory

# arg 1: chromosome number


n=$1

strict_mask=*$n*strict_mask.bed* 
genetic_map=*genetic_map*$n*.txt*
min_size=50


alt=*$n*Altai.vcf.gz
cha=*$n*Chagyrskaya.vcf.gz
den=*$n*Denisova.vcf.gz
vin=*$n*Vindija.vcf.gz


merged_mask=chr"$n"_merged_mask.bed

# archaic merge
make_beds.sh $genetic_map $alt $cha $den $vin
intersect_beds.sh chr"$n"_merged_mask.bed $min_size $strict_mask *$n*Altai*.bed *$n*Chagyrskaya*.bed *$n*Denisova*.bed *$n*Vindija*.bed

bcftools merge -o chr"$n"_archaics.vcf.gz -R $merged_mask $alt $cha $den $vin
bcftools view -o chr"$n"_archaic_alts.vcf.gz -c 1 chr"$n"_archaics.vcf.gz
bcftools index chr"$n"_archaic_alts.vcf.gz


# modern merge
fre=*$n*French-1.vcf.gz
kho=*$n*Khomani_San-2.vcf.gz
han=*$n*Han-1.vcf.gz
pap=*$n*Papuan-2.vcf.gz
yo1=*$n*Yoruba-1.vcf.gz
yo3=*$n*Yoruba-3.vcf.gz

bcftools merge -o chr"$n"_merged.vcf.gz -R $merged_mask -0 chr"$n"_archaic_alts.vcf.gz $fre $kho $han $pap $yo1 $yo3

# rename samples
bcftools reheader -o chr"$n"_merged.vcf.gz -s /c/archaic/data/sample_ids.txt chr"$n"_merged.vcf.gz
