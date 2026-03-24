import random
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def _join_fragments(frag_a, frag_b, bond_type=Chem.BondType.SINGLE):
    """Connect two RDKit mol fragments (each with one dummy * atom) via a bond.

    FragmentOnBonds inserts dummy atoms (atomic number 0) at the cut site.
    This function finds the attachment atom neighbouring the dummy in each
    fragment, combines the two mols, forms a new bond of the specified type,
    then removes the dummy atoms.

    Returns canonical SMILES string, or None if the operation fails.
    """
    dummy_a = [a.GetIdx() for a in frag_a.GetAtoms() if a.GetAtomicNum() == 0]
    dummy_b = [a.GetIdx() for a in frag_b.GetAtoms() if a.GetAtomicNum() == 0]
    if not dummy_a or not dummy_b:
        return None

    neighbors_a = frag_a.GetAtomWithIdx(dummy_a[0]).GetNeighbors()
    neighbors_b = frag_b.GetAtomWithIdx(dummy_b[0]).GetNeighbors()
    if not neighbors_a or not neighbors_b:
        return None

    attach_a = neighbors_a[0].GetIdx()
    attach_b = neighbors_b[0].GetIdx()
    offset = frag_a.GetNumAtoms()

    combined = Chem.CombineMols(frag_a, frag_b)
    rw = Chem.RWMol(combined)
    rw.AddBond(attach_a, attach_b + offset, bond_type)

    # Remove dummy atoms highest-index first to avoid index shifting
    for d in sorted([dummy_a[0], dummy_b[0] + offset], reverse=True):
        rw.RemoveAtom(d)

    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


class MoleculeCrossover:
    """Fragment-based molecular crossover.

    Splits each parent at a random non-ring single bond, then recombines one
    fragment from each parent into a new molecule.  Returns None if no valid
    offspring can be produced (e.g. parents have no rotatable bonds, or the
    joined molecule fails validation).

    Args:
        atom_set: Passed to MoleculeMutator for offspring validation.
            'nlo' (default) or 'drug'.
        max_attempts: How many (cut1, cut2, frag choice) combinations to try
            before giving up.
    """

    def __init__(self, atom_set='nlo', max_attempts=10):
        from molev_utils.molecule_ops import MoleculeMutator
        self.validator = MoleculeMutator(atom_set=atom_set)
        self.allowed_bond_types = self.validator.allowed_bond_types  # [SINGLE, DOUBLE]
        self.max_attempts = max_attempts

    def crossover(self, smiles1: str, smiles2: str) -> 'str | None':
        """Return a crossed offspring SMILES, or None if crossover fails."""
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return None

        # Candidate cut bonds: non-ring single bonds only
        bonds1 = [b.GetIdx() for b in mol1.GetBonds()
                  if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()]
        bonds2 = [b.GetIdx() for b in mol2.GetBonds()
                  if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()]
        if not bonds1 or not bonds2:
            return None

        for _ in range(self.max_attempts):
            cut1 = random.choice(bonds1)
            cut2 = random.choice(bonds2)

            frags1 = Chem.GetMolFrags(Chem.FragmentOnBonds(mol1, [cut1]), asMols=True)
            frags2 = Chem.GetMolFrags(Chem.FragmentOnBonds(mol2, [cut2]), asMols=True)
            if len(frags1) < 2 or len(frags2) < 2:
                continue

            frag_a = random.choice(frags1)
            frag_b = random.choice(frags2)
            join_bond = random.choice(self.allowed_bond_types)

            offspring_smiles = _join_fragments(frag_a, frag_b, join_bond)
            if offspring_smiles and self.validator.validate(offspring_smiles):
                return offspring_smiles

        return None
