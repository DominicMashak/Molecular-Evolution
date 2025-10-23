# Molecular-Evolution

[![CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-EF9421.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) ![Linux](https://img.shields.io/badge/platform-linux-brightgreen)

To learn how to start using, skip to reading [TUTORIAL.md](https://github.com/DominicMashak/Molecular-Evolution/blob/main/TUTORIAL.md)

## Overview

Molecular-Evolution is a GitHub repository for our project of using evolutionary algorithms for optimizing molecular structures, with a primary focus on enhancing Non-Linear Optical (NLO) properties, particularly hyperpolarizability. We utilize genetic and evolutionary optimization algorithms to evolve molecules from initial SMILES representations, evaluating and selecting candidates based on quantum chemical calculations. It is designed for researchers in computational physics/chemistry, materials science who aim to discover novel molecules with improved NLO characteristics.

The project currently implements algorithms such as Simulated Annealing (SA) and Non-dominated Sorting Genetic Algorithm II (NSGA-II). We plan to expand to additional evolutionary algorithms such as Multi-Objective MAP-Elites (MOME). Molecules are represented using SMILES strings generated via RDKit, and properties are computed using a variety of quantum chemistry methods and basis sets. Benchmarking tools are included to compare performance across methods and generate datasets for validation and testing.

## Key Features

- **Evolutionary Optimization**: Apply SA and NSGA-II to evolve molecular populations toward optimal NLO properties.
- **NLO Property Focus**: At the moment, it specifically targets hyperpolarizability (β), along with 
the ability for related properties like polarizability (α), second hyperpolarizability (γ), HUMO-LUMO gap, and total energy to be screened and optimized.
- **SMILES-Based**: Uses RDKit for generating and manipulating SMILES strings, ensuring compatibility with standard cheminformatics pipelines. We currently use 7 mutation operators, which include: changing a bond, adding an atom, adding a branch, deleting an atom, changing an atom, adding a ring, and deleting a ring. All of the mutation operators have mutable weights.

## Supported Quantum Chemistry Methods

The repository supports a range of computational methods for property calculations, allowing users to balance speed and accuracy:

- **Coupled-Cluster Methods**:
  - CCSD(T), CCSD

- **Ab Initio and DFT Methods**:
  - Hartree-Fock (HF)
  - Density Functional Theory (DFT) variants: CAM-B3LYP, B3LYP, PBE, PBE0

- **Semi-Empirical Methods**:
  - MINDO3
  - MOPAC PM7, PM6

- **Tight-Binding Methods**:
  - xTB: GFN0-xTB, GFN1-xTB, GFN2-xTB, GFNFF

- **Supported basis sets (non-xTB methods) include**: 
  - STO-3G, 3-21G, 6-31G, 6-31G*, 6-31G**, 6-31+G*, 6-311G, 6-311G*, 6-311G**, 6-311++G**, def2-SVP, def2-TZVP, def2-TZVPP, cc-pVDZ, cc-pVTZ, cc-pVQZ, aug-cc-pVDZ, aug-cc-pVTZ, def2-TZVP

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

## License

This project is licensed under the Creative Commons BY-NC-ND 4.0 License.

https://creativecommons.org/licenses/by-nc-nd/4.0/

## References

Mashak, D., & Alexander, S. A. Finding Molecules with Large Hyperpolarizabilities. MATCH Commun. Math. Comput. Chem. 94 (2025) 633–644. https://match.pmf.kg.ac.rs/issues/m94n3/m94n3_25824.html

Mashak, D., & Alexander, S. A. Finding Molecules with Specific Properties: Simulated Annealing vs. Evolution. Companion Proceedings of the Genetic and Evolutionary Computation Conference (GECCO ’25). ACM, 2025. https://doi.org/10.1145/3712255.3726635

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

Qiming Sun. (2024). Semiempirical module for PySCF. GitHub. https://github.com/pyscf/semiempirical

Moussa, J. E., & Stewart, J. J. P. (2025). MOPAC (v23.1.2). Zenodo. https://doi.org/10.5281/zenodo.14885238

Landrum, G. (2010). RDKit: Open-source cheminformatics. https://www.rdkit.org

Hourahine, B., Aradi, B., Blum, V., Bonafé, F., Buccheri, A., Camacho, C., Cevallos, C., Deshaye, M. Y., Dumitrică, T., Dominguez, A., Ehlert, S., Elstner, M., van der Heide, T., Hermann, J., Irle, S., Kranz, J. J., Köhler, C., Kowalczyk, T., Kubař, T., Lee, I. S., Lutsker, V., Maurer, R. J., Min, S. K., Mitchell, I., Negre, C., Niehaus, T. A., Niklasson, A. M. N., Page, A. J., Pecchia, A., Penazzi, G., Persson, M. P., Řezáč, J., Sánchez, C. G., Sternberg, M., Stöhr, M., Stuckenberg, F., Tkatchenko, A., Yu, V. W., & Frauenheim, T. (2020). DFTB+, a software package for efficient approximate density functional theory based atomistic simulations. The Journal of Chemical Physics, 152. https://doi.org/10.1063/1.5143190
