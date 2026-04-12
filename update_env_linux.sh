#!/bin/bash
<<<<<<< Updated upstream
# Update existing mol-evo conda environment on Linux with new dependencies
# Run from the Molecular-Evolution project root

set -e
=======
set -euo pipefail
# Update existing mol-evo conda environment on Linux with new dependencies
# Run from the Molecular-Evolution project root

>>>>>>> Stashed changes

ENV_NAME="mol-evo"

echo "======================================================================"
echo "Updating ${ENV_NAME} conda environment (Linux)"
echo "======================================================================"

# Check if environment exists
if ! conda env list | grep -q "${ENV_NAME}"; then
    echo "Environment '${ENV_NAME}' not found. Creating from scratch..."
    conda env create -f environment.yml
    exit 0
fi

echo ""
echo "Installing new dependencies into existing ${ENV_NAME} environment..."
echo ""

# Add pytorch channel if not already present
conda config --env --add channels pytorch 2>/dev/null || true

# Install conda packages
echo "[1/3] Installing conda packages (scikit-learn, pytorch)..."
conda install -n ${ENV_NAME} -c conda-forge -c pytorch \
    scikit-learn \
    pytorch::pytorch \
    --yes

# Install pip packages
echo ""
echo "[2/3] Installing pip packages (transformers, umap-learn)..."
"$HOME/miniconda3/envs/${ENV_NAME}/bin/pip" install \
    transformers \
    umap-learn

# Verify installation
echo ""
echo "[3/3] Verifying installation..."
"$HOME/miniconda3/envs/${ENV_NAME}/bin/python" -c "
import torch
import transformers
from umap import UMAP
from sklearn.decomposition import PCA

print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
print(f'  Transformers: {transformers.__version__}')
print(f'  UMAP:         OK')
print(f'  scikit-learn: OK')

# Test ChemBERTa loading
print()
print('  Loading ChemBERTa-77M-MTR (first run downloads ~300MB)...')
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MTR')
model = AutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MTR')
print('  ChemBERTa:    OK')
"

echo ""
echo "======================================================================"
echo "Environment updated successfully!"
echo "======================================================================"
