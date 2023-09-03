#!/bin/bash -l

for i in `seq 16 29`
do
    cmd="msub -v SEED=${i} run_hebo_spec.moab"
    echo $cmd
    $cmd
done
