<h1>
  <img align="absmiddle" src="mol-evo-logo.png" width="96" height="96" alt="Molecular Evolution logo" style="margin-right: 18px;">
  Molecular-Evolution
</h1>

[![CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-EF9421.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) [![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)](#) [![macOS](https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=F0F0F0)](#) [![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=fff)](#) [![ROCm](https://img.shields.io/badge/ROCm-ED1C24?logo=amd&logoColor=white)](#) [![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](#) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](#)

Please note this is a very work-in-progress repository and project. For inquiries, contact [Dominic Mashak](mailto:mashakd@southwestern.edu)

To learn how to start using, skip to reading [TUTORIAL.md](https://github.com/DominicMashak/Molecular-Evolution/blob/main/TUTORIAL.md)

To cite use [CITATION.cff](https://github.com/DominicMashak/Molecular-Evolution/blob/main/CITATION.cff) or BibTeX [CITATION.bib](https://github.com/DominicMashak/Molecular-Evolution/blob/main/CITATION.bib)

## Overview

Molecular-Evolution is a GitHub repository for our project of using evolutionary algorithms for optimizing molecular structures, with a primary focus on enhancing Non-Linear Optical (NLO) properties, and continued work is in the domain of drug discovery. We use evolutionary optimization algorithms to evolve molecules, evaluating and selecting candidates based on performance evaluation (dependent on the domain). It is designed for researchers in computational physics/chemistry and materials science who aim to discover novel molecules, as well as computer scientists interested in molecular optimization.

## Supported Algorithms

- Simulated Annealing
- Mu + Lambda
- Non-dominated Sorting Genetic Algorithm II (NSGA-II)
- Multi-dimensional Archive of Phenotypic Elites (MAP-Elites)
- Multi-Objective MAP-Elites (MOME)

## Supported Molecular Genotypes

- Simplified Molecular Input Line Entry System (SMILES)
- Self-Referencing Embedded Strings (SELFIES)

## Supported Quantum Chemistry Methods

The repository supports a range of computational methods for property calculations, allowing users to balance speed and accuracy:

- **Coupled-Cluster Methods**:
  - CCSD(T), CCSD

- **Ab Initio and DFT Methods**:
  - Hartree-Fock (HF)
  - Density Functional Theory (DFT): CAM-B3LYP, B3LYP, PBE, PBE0, M06-2X

- **Semi-Empirical Methods**:
  - MINDO3
  - MOPAC PM7, PM6

- **Tight-Binding Methods**:
  - xTB: GFN0-xTB, GFN1-xTB, GFN2-xTB, GFNFF

- **Supported basis sets**: 
  - STO-3G, 3-21G, 6-31G, 6-31G*, 6-31G**, 6-31+G*, 6-311G, 6-311G*, 6-311G**, 6-311++G**, def2-SVP, def2-TZVP, def2-TZVPP, cc-pVDZ, cc-pVTZ, cc-pVQZ, aug-cc-pVDZ, aug-cc-pVTZ

These methods enable rapid screening for evolutionary optimization or more accurate but computationally expensive calculations.

## Molecular Restrictions

Default constraints vary by domain.

**NLO domain** constraints are applied to ensure chemically feasible small conjugated molecules:

- Maximum number of atoms: 30 (excluding implicit hydrogens)
- Allowed atoms: C, N, O (with implicit H)
- Valence rules: No more than 4 bonds per carbon atom
- Bond types: No triple bonds
- Ring structures: No 3- or 4-membered rings; 5- and 6-membered rings are permitted
- Conjugation rules: No atom with two or more double bonds in the same ring

**Drug domain** (`--atom-set drug`) uses a broader atom set appropriate for drug-like molecules, with relaxed size and bond type constraints.

All restrictions can be modified for specific use.

## License

This project is licensed under the Creative Commons BY-NC 4.0 License.

[https://creativecommons.org/licenses/by-nc-nd/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)

## References

Mashak, D., Schrum, J., & Alexander, S. A. (2026).
Multi-Objective Evolutionary Design of Molecules with Enhanced Nonlinear Optical Properties.
*arXiv preprint arXiv:2602.16044* [physics.comp-ph].
https://arxiv.org/abs/2602.16044

Mashak, D., & Alexander, S. A. (2025).
Benchmarking Hartree-Fock and DFT for Molecular Hyperpolarizability: Implications for Evolutionary Design.
*arXiv preprint arXiv:2511.17767* [physics.chem-ph].
https://arxiv.org/abs/2511.17767

Mashak, D., & Alexander, S. A. (2025).
Finding Molecules with Specific Properties: Simulated Annealing vs. Evolution.
*Companion Proceedings of the Genetic and Evolutionary Computation Conference (GECCO ‘25)*. ACM.
https://doi.org/10.1145/3712255.3726635

Mashak, D., & Alexander, S. A. (2025).
Finding Molecules with Large Hyperpolarizabilities.
*MATCH Communications in Mathematical and in Computer Chemistry*, 94, 633–644.
https://match.pmf.kg.ac.rs/issues/m94n3/m94n3_25824.html

Moussa, J. E., & Stewart, J. J. P. (2025).
MOPAC (v23.1.2). Zenodo.
https://doi.org/10.5281/zenodo.14885238

Sun, Q. (2024).
Semiempirical module for PySCF. GitHub.
https://github.com/pyscf/semiempirical

Ahmad, W., Simon, E., Chithrananda, S., Grand, G., & Ramsundar, B. (2022).
ChemBERTa-2: Towards chemical foundation models.
*arXiv preprint arXiv:2209.01712*.
https://arxiv.org/abs/2209.01712

Gao, W., Raghavan, K., Coley, C. W., & Gomes, C. P. (2022).
Sample efficiency matters: a benchmark for practical molecular optimization.
*Advances in Neural Information Processing Systems*, 35.
https://arxiv.org/abs/2206.12411

Pierrot, T., Grillotti, L., Bernin, L., & Cully, A. (2022).
Multi-objective quality diversity optimization.
*Proceedings of the Genetic and Evolutionary Computation Conference (GECCO ‘22)*. ACM.
https://doi.org/10.1145/3512290.3528823

Krenn, M., Häse, F., Nigam, A., Friederich, P., & Aspuru-Guzik, A. (2020).
Self-referencing embedded strings (SELFIES): A 100% robust molecular string representation.
*Machine Learning: Science and Technology*, 1(4), 045024.
https://doi.org/10.1088/2632-2153/aba947

Sun, Q., Berkelbach, T. C., Blunt, N. S., Booth, G. H., Guo, S., Li, Z., Liu, J., McClain, J. D., Sayfutyarova, E. R., Sharma, S., Wouters, S., & Chan, G. K.-L. (2018).
The Python-based Simulations of Chemistry Framework (PySCF).
*Wiley Interdisciplinary Reviews: Computational Molecular Science*, 8(1), e1340.
https://doi.org/10.1002/wcms.1340

Mouret, J.-B., & Clune, J. (2015).
Illuminating search spaces by mapping elites.
*arXiv preprint arXiv:1504.04909*.
https://arxiv.org/abs/1504.04909

Koes, D. R., Baumgartner, M. P., & Camacho, C. J. (2013).
Lessons learned in empirical scoring with smina from the CSAR 2011 benchmarking exercise.
*Journal of Chemical Information and Modeling*, 53(8), 1893–1904.
https://doi.org/10.1021/ci300604z

Stewart, J. J. P. (2013).
Optimization of parameters for semiempirical methods VI: more modifications to the NDDO approximations and re-optimization of parameters.
*Journal of Molecular Modeling*, 19(1), 1–32.
https://doi.org/10.1007/s00894-012-1667-x

Bickerton, G. R., Paolini, G. V., Besnard, J., Muresan, S., & Hopkins, A. L. (2012).
Quantifying the chemical beauty of drugs.
*Nature Chemistry*, 4(2), 90–98.
https://doi.org/10.1038/nchem.1243

Landrum, G. (2010).
RDKit: Open-source cheminformatics.
https://www.rdkit.org

Ertl, P., & Schuffenhauer, A. (2009).
Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions.
*Journal of Cheminformatics*, 1(8).
https://doi.org/10.1186/1758-2946-1-8

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
A fast and elitist multiobjective genetic algorithm: NSGA-II.
*IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
https://doi.org/10.1109/4235.996017

Beyer, H.-G., & Schwefel, H.-P. (2002).
Evolution strategies – A comprehensive introduction.
*Natural Computing*, 1(1), 3–52.
https://doi.org/10.1023/A:1015059928466

Lipinski, C. A., Lombardo, F., Dominy, B. W., & Feeney, P. J. (2001).
Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings.
*Advanced Drug Delivery Reviews*, 46(1–3), 3–26.
https://doi.org/10.1016/S0169-409X(00)00129-0

Du, Q., Faber, V., & Gunzburger, M. (1999).
Centroidal Voronoi tessellations: Applications and algorithms.
*SIAM Review*, 41(4), 637–676.
https://doi.org/10.1137/S0036144599352836

Stewart, J. J. P. (1989).
Optimization of parameters for semiempirical methods I. Method.
*Journal of Computational Chemistry*, 10(2), 209–220.
https://doi.org/10.1002/jcc.540100208

Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983).
Optimization by simulated annealing.
*Science*, 220(4598), 671–680.
https://doi.org/10.1126/science.220.4598.671
