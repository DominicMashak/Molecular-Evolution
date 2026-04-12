#!/bin/bash
<<<<<<< Updated upstream
=======
set -euo pipefail
>>>>>>> Stashed changes
# Setup script for molecular docking environment
# This creates a dedicated conda environment with all docking dependencies

set -e  # Exit on error

echo "======================================================================="
echo "Molecular Docking Environment Setup"
echo "======================================================================="
echo ""
echo "This script will:"
echo "  1. Create a new conda environment 'molev-docking' with Python 3.10"
echo "  2. Install all required dependencies (RDKit, Smina, XTB, OpenBabel)"
echo "  3. Install SmartCADD package in editable mode"
echo "  4. Run validation to verify the setup"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 1
fi

echo ""
echo "======================================================================="
echo "Step 1: Creating conda environment from environment-docking.yml"
echo "======================================================================="

# Check if environment already exists
if conda env list | grep -q "molev-docking"; then
    echo ""
    echo "WARNING: Environment 'molev-docking' already exists."
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n molev-docking -y
    else
        echo "Keeping existing environment. Updating instead..."
        conda env update -n molev-docking -f environment-docking.yml --prune
        echo ""
        echo "Environment updated. Proceeding to SmartCADD installation..."
        echo ""
    fi
fi

# Create or update environment
if ! conda env list | grep -q "molev-docking"; then
    echo "Creating new environment (this may take 5-10 minutes)..."
    conda env create -f environment-docking.yml
fi

echo ""
echo "======================================================================="
echo "Step 2: Installing SmartCADD package"
echo "======================================================================="

# Check if SmartCADD directory exists
if [ ! -d "$HOME/SmartCADD" ]; then
    echo ""
    echo "ERROR: SmartCADD directory not found at ~/SmartCADD"
    echo ""
    echo "Please clone SmartCADD first:"
    echo "  cd ~"
    echo "  git clone https://github.com/your-smartcadd-repo/SmartCADD.git"
    echo ""
    exit 1
fi

# Activate environment and install SmartCADD
echo "Installing SmartCADD in editable mode..."
eval "$(conda shell.bash hook)"
conda activate molev-docking

cd "$HOME/SmartCADD"
pip install -e .

echo ""
echo "======================================================================="
echo "Step 3: Validating installation"
echo "======================================================================="

cd "$HOME/Molecular-Evolution"
python tools/validate_docking_env.py --test-all

VALIDATION_STATUS=$?

echo ""
echo "======================================================================="
echo "Setup Complete!"
echo "======================================================================="

if [ $VALIDATION_STATUS -eq 0 ]; then
    echo ""
    echo "✓ SUCCESS: All dependencies installed and validated!"
    echo ""
    echo "To use the docking environment:"
    echo "  conda activate molev-docking"
    echo "  cd ~/Molecular-Evolution"
    echo "  ./algorithims/nsga2/run_nsga2_drug.sh 42 docking 1AQ1"
    echo ""
else
    echo ""
    echo "⚠ WARNING: Some validation checks failed."
    echo ""
    echo "Please review the error messages above."
    echo "You may still be able to use descriptor mode:"
    echo "  conda activate molev-docking"
    echo "  ./algorithims/nsga2/run_nsga2_drug.sh 42 descriptors"
    echo ""
    echo "For troubleshooting, see: DOCKING_SETUP_GUIDE.md"
    echo ""
fi

echo "Deactivating environment..."
conda deactivate

echo ""
echo "Next steps:"
echo "  1. Activate: conda activate molev-docking"
echo "  2. Validate: python tools/validate_docking_env.py --test-all"
echo "  3. Run test: ./algorithims/nsga2/run_nsga2_drug.sh 42 docking 1AQ1"
echo ""
