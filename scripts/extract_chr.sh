#!/bin/bash

FILENAMES=()

while [ $# -gt 0 ];
do
	case $1 in
		-c | --chrom)
			CHROM=$2
			shift
			;;
		*)
			if [[ $1 == *.vcf.gz ]]
			then	
				FILENAMES+=($1)
			else
				echo "A non-.vcf.gz file was passed as an argument! ${1}"
			fi
			;;
	esac
	shift
done


for FILENAME in ${FILENAMES[@]};
do
	if ! test -f "$FILENAME.csi";
	then
		bcftools index $FILENAME
	fi
	NEWFILENAME="chr${CHROM}_${FILENAME}"
	bcftools view -r $CHROM -o $NEWFILENAME $FILENAME
done
