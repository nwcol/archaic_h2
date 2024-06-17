#!/bin/bash


boot=$1
graph=$2
params=$3
out=$4
temp=x.yaml

iter=100
verb=10
u=1.35e-8

python -m archaic.scripts.permute_graph -g $graph -p $params -o $temp

python -m archaic.scripts.infer -d $boot -g $temp -p $params -o $out -m $iter -v $verb -u $u

rm $temp
