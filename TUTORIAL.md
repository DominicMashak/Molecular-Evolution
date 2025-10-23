# Molecular Evolution - Setup Tutorial

This guide will walk you through the complete setup and testing process for the Molecular Evolution project.

## Prerequisites

- Linux system (Ubuntu/Debian recommended)
- System that meets [SYSTEM_REQUIREMENTS.txt](https://github.com/DominicMashak/Molecular-Evolution/blob/main/SYSTEM_REQUIREMENTS.txt)

## Step 1: Install Git

Update your system and install Git:

```bash
sudo apt update
sudo apt install git
```

## Step 2: GitHub Access Token

1. Navigate to [GitHub Token Settings](https://github.com/settings/tokens)
2. Click **"Generate new token (classic)"**
3. Under **"Select scopes"**, check **"repo"**
4. Set the **"Note"** field to: `Molecular-Evolution`
5. Click **"Generate token"**
6. **Copy the TOKEN** (you'll need it in the next step)

## Step 3: Clone Repository

Replace `TOKEN` with your actual GitHub token:

```bash
git clone https://TOKEN@github.com/DominicMashak/Molecular-Evolution
cd Molecular-Evolution
```

## Step 4: Install Miniconda

Download and install Miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**Note:** Say `yes` to all installation prompts.

After installation completes, activate the changes:

```bash
source ~/.bashrc
```

Verify the installation:

```bash
conda --version
```

Ensure conda is in your PATH:

```bash
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
```

## Step 5: Create Conda Environment

Create and activate the project environment:

```bash
conda env create -f environment.yml
conda activate mol-evo
```

## Step 6: Test 1 - DFT Quantum Chemistry Calculation

Navigate to the quantum chemistry directory and run the test:

```bash
cd quantum_chemistry
bash run_dft.sh
```

### Expected Results

The calculation should complete in approximately **10 minutes or less**. Verify your results match:

- **Beta mean result:** `6.101466e+03` or `6,101.47 a.u.`
- **HOMO-LUMO gap:** `3.767602 eV`
- **Total energy:** `-491.949483 a.u.`

If your results match, the DFT module is working correctly!

## Step 7: Test 2 - NSGA-II Algorithm

Navigate to the NSGA-II directory and run the test:

```bash
cd ..
cd algorithms/nsga2
bash run_nsga2.sh
```

### Expected Results

After **Generation 1** completes, check the `nsga2_results` folder for a generated graph.

If a graph appears in the results folder, the NSGA-II algorithm is working correctly!

## What now?

With both tests passing, you're ready to start using Molecular-Evolution! Explore the repository for additional examples and documentation.

