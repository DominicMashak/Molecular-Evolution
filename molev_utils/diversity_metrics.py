"""
Molecular diversity metrics for evaluating archive quality.

All metrics are standard in the PMO / MOSES benchmarking literature and
are domain-agnostic (work for any SMILES-representable molecule).

Functions
---------
internal_diversity  : mean pairwise Tanimoto distance (IntDiv, PMO standard)
scaffold_count      : number of unique Bemis-Murcko scaffolds
novelty             : fraction of solutions not in a reference set
compute_diversity_metrics : compute all metrics and return a single dict
"""

from typing import List, Optional, Set
import numpy as np


def internal_diversity(
    smiles_list: List[str],
    radius: int = 2,
    nbits: int = 2048,
    max_sample: int = 1000,
) -> float:
    """Mean pairwise Tanimoto distance over Morgan (ECFP4) fingerprints.

    Returns a value in [0, 1].  Higher means more chemically diverse.
    Samples at most *max_sample* molecules when the list is larger to keep
    computation tractable (O(n²) pairwise similarity).

    Uses RDKit's BulkTanimotoSimilarity which is vectorised in C++; computing
    1 000 × 999 / 2 ≈ 500 K pairs takes <1 ms.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    fps = []
    for s in smiles_list:
        if s is None:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fps.append(
                AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            )

    if len(fps) < 2:
        return 0.0

    if len(fps) > max_sample:
        idx = np.random.choice(len(fps), max_sample, replace=False)
        fps = [fps[i] for i in idx]

    n = len(fps)
    total = 0.0
    count = 0
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        total += sum(1.0 - s for s in sims)
        count += len(sims)

    return total / count if count > 0 else 0.0


def scaffold_count(smiles_list: List[str]) -> int:
    """Number of unique Bemis-Murcko scaffolds in the list.

    Scaffolds are computed without chirality so enantiomers share a scaffold.
    Molecules that fail scaffold extraction are silently skipped.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds: Set[str] = set()
    for s in smiles_list:
        if s is None:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
            scaffolds.add(scaf)
        except Exception:
            pass

    return len(scaffolds)


def novelty(smiles_list: List[str], reference_set: Set[str]) -> float:
    """Fraction of non-None SMILES not present in *reference_set*.

    Returns a float in [0, 1].  1.0 means all solutions are novel.
    """
    valid = [s for s in smiles_list if s is not None]
    if not valid:
        return 0.0
    return sum(1 for s in valid if s not in reference_set) / len(valid)


def compute_diversity_metrics(
    smiles_list: List[str],
    reference_set: Optional[Set[str]] = None,
    max_sample: int = 1000,
) -> dict:
    """Compute all diversity metrics and return a logging-ready dict.

    Parameters
    ----------
    smiles_list :
        SMILES strings of the solutions to analyse (e.g. all archive members).
    reference_set :
        Optional set of SMILES (training molecules, seed set) used to compute
        novelty.  If None, the 'novelty' key is omitted from the result.
    max_sample :
        Maximum solutions used for the O(n²) IntDiv computation.

    Returns
    -------
    dict with keys:
        int_div        : float  — mean pairwise Tanimoto distance
        scaffold_count : int    — unique Bemis-Murcko scaffolds
        n_unique       : int    — unique SMILES in smiles_list
        novelty        : float  — (only if reference_set provided)
    """
    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for s in smiles_list:
        if s is not None and s not in seen:
            seen.add(s)
            unique.append(s)

    metrics: dict = {
        'int_div': internal_diversity(unique, max_sample=max_sample),
        'scaffold_count': scaffold_count(unique),
        'n_unique': len(unique),
    }
    if reference_set is not None:
        metrics['novelty'] = novelty(unique, reference_set)

    return metrics
