#!/bin/bash


IN=$1
OUT=$2
STARTDIR=$PWD


INDIR=$(dirname $IN)
cd $INDIR
INPATH=$PWD
FILEPATH="${INPATH}/$(basename $IN)"
cd $STARTDIR


OUTDIR=$(dirname $OUT)
if [[ $OUTDIR == . ]]; then
    echo "OUTPATH ${OUTPATH}"
else
    cd $OUTDIR
    OUTPATH=$PWD
    cd $STARTDIR
fi
OUT="${OUTPATH}/$(basename $OUT)"


python -m util.scripts.make_map_bed_file $FILEPATH $OUT

