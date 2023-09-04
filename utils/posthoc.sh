#!/bin/bash -l

for cmd in "./utils/remove.sh" "./utils/compress.sh" "./utils/clean.sh"
do
    echo $cmd
    $cmd
done
