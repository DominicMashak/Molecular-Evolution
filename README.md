# Molecular-Evolution

[![CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-EF9421.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)

## Overview

Molecular-Evolution is a GitHub repository for our project of using evolutionary algorithms for optimizing molecular structures, with a primary focus on enhancing Non-Linear Optical (NLO) properties, particularly hyperpolarizability. We utilize genetic and evolutionary optimization algorithms to evolve molecules from initial SMILES representations, evaluating and selecting candidates based on quantum chemical calculations. It is designed for researchers in computational physics/chemistry, materials science who aim to discover novel molecules with improved NLO characteristics.

The project currently implements algorithms such as Simulated Annealing (SA) and Non-dominated Sorting Genetic Algorithm II (NSGA-II). We plan to expand to additional evolutionary algorithms such as Multi-Objective MAP-Elites (MOME). Molecules are represented using SMILES strings generated via RDKit, and properties are computed using a variety of quantum chemistry methods and basis sets. Benchmarking tools are included to compare performance across methods and generate datasets for validation and testing.

## Key Features

- **Evolutionary Optimization**: Apply SA and NSGA-II to evolve molecular populations toward optimal NLO properties.
- **NLO Property Focus**: At the moment specifically targets hyperpolarizability (β), along with 
the ability for related properties like polarizability (α) and second hyperpolarizability (γ), to be screened and optimized.
- **SMILES-Based**: Uses RDKit for generating and manipulating SMILES strings, ensuring compatibility with standard cheminformatics pipelines. We currently use 7 mutation operators, which include: changing a bond, adding an atom, adding a branch, deleting an atom, changing an atom, adding a ring, and deleting a ring. All of the mutation operators have mutable weights.
- **Benchmarking and Statistics**: Includes tools for generating benchmark datasets and computing various statistical comparisons (e.g., mean, max, min values for properties like β, HOMO-LUMO, and Kendall Tau, and Spearman Rho for rank correlation). Visualization of results and method comparisons is also included.
- **Dataset Generation**: Integrates with this Hugging Face dataset at [maykcaldas/smiles-transformers](https://huggingface.co/datasets/maykcaldas/smiles-transformers) to create custom benchmark sets.

## Supported Quantum Chemistry Methods

The repository supports a range of computational methods for property calculations, allowing users to balance speed and accuracy:

- **Ab Initio and DFT Methods**:
  - Hartree-Fock (HF)
  - Density Functional Theory (DFT) variants: B3LYP, PBE, PBE0

- **Semi-Empirical Methods**:
  - PM3, AM1, MNDO
  - MOPAC PM7

- **Tight-Binding Methods**:
  - xTB: GFN0-xTB, GFN1-xTB, GFN2-xTB, GFNFF

Supported basis sets for non-xTB methods include: STO-3G, 3-21G, 6-31G, 6-31G*, 6-311G.

These methods enable rapid screening (e.g., via xTB) to more accurate but computationally expensive calculations (e.g., DFT with larger basis sets).

## Molecular Restrictions

To ensure chemically feasible molecules, the following default constraints are applied during evolution and generation:

- Maximum number of atoms: 50 (excluding implicit hydrogens)
- Allowed atoms: C, N, O (with implicit H)
- Valence rules: No more than 4 bonds per carbon atom
- Bond types: No triple bonds
- Ring structures: No 3- or 4-membered rings; 5- and 6-membered rings are permitted
- Conjugation rules: No atom with two or more double bonds in the same ring

These restrictions can be modified for specific use.

## Installation

TODO

## Usage

TODO

## Contributing

TODO

## License

This project is licensed under the Creative Commons BY-NC-ND 4.0 License.

https://creativecommons.org/licenses/by-nc-nd/4.0/

## References

Dominic Mashak and S.A. Alexander, Finding Molecules with Large Hyperpolarizabilities. MATCH Commun. Math. Comput. Chem. 94 (2025) 633-644. https://doi.org/10.46793/match94-3.25824

Steven Alexander and Dominic Mashak. 2025. Finding Molecules with Specific Properties: Simulated Annealing vs. Evolution. In Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '25 Companion). Association for Computing Machinery, New York, NY, USA, 759–762. https://doi.org/10.1145/3712255.3726635

Sun, Q., Berkelbach, T. C., Blunt, N. S., Booth, G. H., Guo, S., Li, Z., Liu, J., McClain, J. D., Sayfutyarova, E. R., Sharma, S., Wouters, S., & Chan, G. K.-L. (2018). 
The Python-based Simulations of Chemistry Framework (PySCF). 
*Wiley Interdisciplinary Reviews: Computational Molecular Science*, 8(1), e1340. 
https://doi.org/10.1002/wcms.1340

Stewart, J. J. P. (2013). 
Optimization of parameters for semiempirical methods VI: more modifications to the NDDO approximations and re-optimization of parameters. 
*Journal of Molecular Modeling*, 19(1), 1-32. 
https://doi.org/10.1007/s00894-012-1667-x

Stewart, J. J. P. (1989). 
Optimization of parameters for semiempirical methods I. Method. 
*Journal of Computational Chemistry*, 10(2), 209-220. 
https://doi.org/10.1002/jcc.540100208

Caldas, M. (2023). 
SMILES-transformers Dataset. 
Hugging Face. 
https://huggingface.co/datasets/maykcaldas/smiles-transformers

sunqm. (2024). Semiempirical module for PySCF. GitHub. https://github.com/pyscf/semiempirical
