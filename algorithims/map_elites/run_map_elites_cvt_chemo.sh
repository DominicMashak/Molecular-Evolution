#!/bin/bash
SEED=${1:-42}
cell_line=${2:-22RV1}
qed_min=${3:-0.3}
filter_lipinski=${4:-true}
sa_max=${5:-6.0}
initial_molecules_file=${6:-$HOME/Documents/GitHub/Molecular-Evolution/drug/initial_molecules.txt}


echo "======================================================================"
echo "Running MAP-Elites CVT for GPDRP with seed ${SEED} cell line ${cell_line}"
echo "======================================================================"

cd ~/Documents/GitHub/Molecular-Evolution/algorithims/map_elites

export PYTHONPATH="$HOME/Documents/GitHub/Molecular-Evolution:$PYTHONPATH"

python main.py \
    --fitness-mode gpdrp \
    --cell-line ${cell_line} \
    --objective-key lnic50 \
    --qed-min ${qed_min} \
    --sa-max ${sa_max} \
    $( [ "$filter_lipinski" = "true" ] && echo "--filter-lipinski" ) \
    --archive-type cvt \
    --n-centroids 100 \
    --pop_size 20 \
    --n_gen 5 \
    --save_frequency 1 \
    --log_frequency 1 \
    --output_dir map_elites_gpdrp_results_seed_${SEED} \
    --seed ${SEED} \
    --atom-set gpdrp \
    --initial-population-file ${initial_molecules_file} \
    --verbose

echo "======================================================================"
echo "MAP-Elites CVT complete! Results in map_elites_gpdrp_results_seed_${SEED}/"
echo "======================================================================"