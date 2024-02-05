#!/bin/bash


FILES=()
STARTDIR=$PWD
OUTDIR="."


while [ $# -gt 0 ]; do
	case $1 in
		-o | --out)
			OUTDIR=$2
			shift
			;;
		*)
			if [[ $1 == *.vcf.gz ]]
			then
				FILES+=($1)
			else
				echo "A non-.vcf.gz file format was passed as an argument: ${1}"
			fi
			;;
	esac
	shift
done


for FILENAME in ${FILES[@]};
do
	DIR=$(dirname $FILENAME)
	cd $DIR
	INPATH=$PWD
	cd $STARTDIR
	cd $OUTDIR
	OUTPATH=$PWD
	STEM="$(basename ${FILENAME%%.*})"
	IN="${INPATH}/${STEM}.vcf.gz"
	OUT="${OUTPATH}/${STEM}_variants.vcf.gz"
	bcftools view -c 1 -o $OUT $IN
	cd $STARTDIR
done

