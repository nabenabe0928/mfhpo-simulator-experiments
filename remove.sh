#!/bin/bash -l

module load tools/singularity/3.11
singularity exec mfhpo-simulator.sif python -m src.remove_files
