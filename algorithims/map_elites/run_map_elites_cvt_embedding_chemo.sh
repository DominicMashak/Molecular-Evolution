#!/bin/bash
# Run CVT-MAP-Elites with ChemBERTa-2 embeddings for GPDRP drug-design optimization
# Uses transformer-based molecular embeddings as CVT behavior descriptors

SEED=${1:-42}
cell_line=${2:-22RV1}
qed_min=${3:-0.1}
filter_lipinski=${4:-true}
sa_max=${5:-6.5}
initial_molecules_file=${6:-$HOME/Documents/GitHub/Molecular-Evolution/drug/initial_molecules_gdsc2.txt}
inference_mode=${7:-average}
gpdrp_dir=${8:-$HOME/Documents/GitHub/GPDRP}
n_centroids=${9:-100}
embedding_dims=${10:-8}
pop_size=${11:-50}
n_gen=${12:-100}
iterations_per_gen=${13:-5}

echo "======================================================================"
echo "Running MAP-Elites CVT (ChemBERTa embeddings) for GPDRP"
echo "  Seed: ${SEED}"
echo "  Cell line: ${cell_line}"
echo "  GPDRP dir: ${gpdrp_dir}"
echo "  CVT centroids: ${n_centroids}"
echo "  Embedding dims: ${embedding_dims}"
echo "  Pop size: ${pop_size}  N gen: ${n_gen}  Iter/gen: ${iterations_per_gen}"
echo "======================================================================"

cd ~/Documents/GitHub/Molecular-Evolution/algorithims/map_elites

export PYTHONPATH="$HOME/Documents/GitHub/Molecular-Evolution:$PYTHONPATH"

# Required to avoid OpenMP/numba threading conflicts that cause
# segfaults when ChemBERTa (transformers) and UMAP (numba) are
# loaded in the same process on macOS, especially combined with
# the GPDRP subprocess calls in gpdrp_interface.py
export NUMBA_THREADING_LAYER=workqueue
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

if [ "$filter_lipinski" = "true" ]; then
    LIPINSKI_FLAG="--filter-lipinski"
else
    LIPINSKI_FLAG=""
fi

python main.py \
    --fitness-mode gpdrp \
    --cell-line ${cell_line} \
    --inference-mode ${inference_mode} \
    --objective-key lnic50 \
    --qed-min ${qed_min} \
    --sa-max ${sa_max} \
    ${LIPINSKI_FLAG} \
    --atom-set gpdrp \
    --gpdrp-dir ${gpdrp_dir} \
    --initial-population-file ${initial_molecules_file} \
    --archive-type cvt \
    --n-centroids ${n_centroids} \
    --cvt-samples 50000 \
    --cvt-measures embedding \
    --embedding-model DeepChem/ChemBERTa-77M-MTR \
    --embedding-dims ${embedding_dims} \
    --embedding-device cpu \
    --embedding-sample-size 1000 \
    --pop_size ${pop_size} \
    --n_gen ${n_gen} \
    --iterations_per_gen ${iterations_per_gen} \
    --save_frequency 10 \
    --log_frequency 5 \
    --output_dir map_elites_embedding_gpdrp_results_seed_${SEED} \
    --seed ${SEED} \
    --verbose

echo "======================================================================"
echo "MAP-Elites CVT (ChemBERTa embeddings) complete! Results in:"
echo "  map_elites_embedding_gpdrp_results_seed_${SEED}/"
echo "======================================================================"