from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
import random

# Suppress RDKit warnings for invalid molecules (they get rejected anyway)
RDLogger.DisableLog('rdApp.*')

class MoleculeMutator:
    """Perform mutations on molecules represented as SMILES strings.

    Supports bond-type changes, atom additions/removals, atom-type changes,
    ring additions/removals, and validation. Mutations are restricted to
    allowed atoms and allowed bond types (single, double).

    Args:
        atom_set: Which atom set to use. "nlo" for C/N/O only (original),
            "drug" for C/N/O/S/F/Cl/Br (drug-like molecules).
    """

    # Predefined atom sets
    ATOM_SETS = {
        'nlo': ['C', 'N', 'O'],
        'drug': ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br'],
    }

    # Atoms that can form at least 2 bonds (for insertion mutations)
    MULTIVALENT_ATOMS = {
        'nlo': ['C', 'N', 'O'],
        'drug': ['C', 'N', 'O', 'S'],  # Excludes F, Cl, Br (only 1 bond)
    }

    # Max valence for each atom (used to check compatibility)
    MAX_VALENCE = {'C': 4, 'N': 3, 'O': 2, 'S': 2, 'F': 1, 'Cl': 1, 'Br': 1}

    # Atomic numbers for validation
    ATOMIC_NUMBERS = {
        'nlo': {1, 6, 7, 8},           # H, C, N, O
        'drug': {1, 6, 7, 8, 9, 16, 17, 35},  # H, C, N, O, F, S, Cl, Br
    }

    def __init__(self, atom_set='nlo'):
        """Initialize allowed atoms and bond types."""
        if atom_set not in self.ATOM_SETS:
            raise ValueError(f"Unknown atom_set '{atom_set}'. "
                             f"Must be one of: {list(self.ATOM_SETS.keys())}")
        self.atom_set = atom_set
        self.allowed_atoms = self.ATOM_SETS[atom_set]
        self.allowed_atomic_numbers = self.ATOMIC_NUMBERS[atom_set]
        self.allowed_bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]
        self.multivalent_atoms = self.MULTIVALENT_ATOMS[atom_set]
    
    def mutate(self, smiles, mutation_type):
        mol = Chem.MolFromSmiles(smiles)
        if not mol or len(Chem.GetMolFrags(mol)) > 1:
            return None
        rw = Chem.RWMol(mol)
        
        # Mutation logic
        if mutation_type == 1:  # Change a bond type
            bonds = [b for b in rw.GetBonds()]
            if bonds:
                bond = random.choice(bonds)
                current_type = bond.GetBondType()
                possible_types = [t for t in self.allowed_bond_types if t != current_type]
                if possible_types:
                    new_type = random.choice(possible_types)
                    new_rw = Chem.RWMol(rw)
                    new_rw.GetBondBetweenAtoms(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()).SetBondType(new_type)
                    # Convert RWMol to SMILES
                    try:
                        new_mol = new_rw.GetMol()
                        Chem.SanitizeMol(new_mol)
                        return Chem.MolToSmiles(new_mol)
                    except:
                        return None
        
        elif mutation_type == 2:  # Add a random atom with a random bond
            bonds = [b for b in rw.GetBonds()]
            if bonds:
                bond = random.choice(bonds)
                a1_idx = bond.GetBeginAtomIdx()
                a2_idx = bond.GetEndAtomIdx()
                # Use only multivalent atoms (need at least 2 bonds for insertion)
                new_atom_symbol = random.choice(self.multivalent_atoms)
                new_atom = Chem.Atom(new_atom_symbol)
                new_rw = Chem.RWMol(rw)
                new_idx = new_rw.AddAtom(new_atom)
                bond_type1 = random.choice(self.allowed_bond_types)
                bond_type2 = random.choice(self.allowed_bond_types)
                new_rw.RemoveBond(a1_idx, a2_idx)
                new_rw.AddBond(a1_idx, new_idx, bond_type1)
                new_rw.AddBond(new_idx, a2_idx, bond_type2)
                # Convert RWMol to SMILES
                try:
                    new_mol = new_rw.GetMol()
                    Chem.SanitizeMol(new_mol)
                    return Chem.MolToSmiles(new_mol)
                except:
                    return None
        
        elif mutation_type == 3:  # Add a random atom as a branch
            atoms = [a for a in rw.GetAtoms()]
            if atoms:
                atom = random.choice(atoms)
                new_atom_symbol = random.choice(self.allowed_atoms)
                new_atom = Chem.Atom(new_atom_symbol)
                new_rw = Chem.RWMol(rw)
                new_idx = new_rw.AddAtom(new_atom)
                bond_type = random.choice(self.allowed_bond_types)
                new_rw.AddBond(atom.GetIdx(), new_idx, bond_type)
                # Convert RWMol to SMILES
                try:
                    new_mol = new_rw.GetMol()
                    Chem.SanitizeMol(new_mol)
                    return Chem.MolToSmiles(new_mol)
                except:
                    return None
        
        elif mutation_type == 4:  # Delete a random atom and its bond
            atoms = [a for a in rw.GetAtoms() if a.GetSymbol() != 'H']
            if len(atoms) <= 1:
                return None
            atom = random.choice(atoms)
            new_rw = Chem.RWMol(rw)
            new_rw.RemoveAtom(atom.GetIdx())
            # Convert RWMol to SMILES
            try:
                new_mol = new_rw.GetMol()
                Chem.SanitizeMol(new_mol)
                return Chem.MolToSmiles(new_mol)
            except:
                return None
        
        elif mutation_type == 5:  # Change an atom type
            atoms = [a for a in rw.GetAtoms() if a.GetSymbol() != 'H']
            if atoms:
                atom = random.choice(atoms)
                current_symbol = atom.GetSymbol()
                # Count bonds on this atom to filter compatible replacements
                num_bonds = sum(int(b.GetBondTypeAsDouble()) for b in atom.GetBonds())
                # Only allow atoms with enough valence capacity
                possible_symbols = [s for s in self.allowed_atoms
                                    if s != current_symbol and self.MAX_VALENCE.get(s, 4) >= num_bonds]
                if possible_symbols:
                    new_symbol = random.choice(possible_symbols)
                    new_rw = Chem.RWMol(rw)
                    new_rw.GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(new_symbol))
                    # Convert RWMol to SMILES
                    try:
                        new_mol = new_rw.GetMol()
                        Chem.SanitizeMol(new_mol)
                        return Chem.MolToSmiles(new_mol)
                    except:
                        return None
        
        elif mutation_type == 6:  # Add a ring
            num_atoms = rw.GetNumAtoms()
            if num_atoms >= 3:
                idx1, idx2 = random.sample(range(num_atoms), 2)
                if rw.GetBondBetweenAtoms(idx1, idx2) is None:
                    bond_type = random.choice(self.allowed_bond_types)
                    new_rw = Chem.RWMol(rw)
                    new_rw.AddBond(idx1, idx2, bond_type)
                    # Convert RWMol to SMILES
                    try:
                        new_mol = new_rw.GetMol()
                        Chem.SanitizeMol(new_mol)
                        return Chem.MolToSmiles(new_mol)
                    except:
                        return None
        
        elif mutation_type == 7:  # Delete a ring
            ring_bonds = [b for b in rw.GetBonds() if b.IsInRing()]
            if ring_bonds:
                bond = random.choice(ring_bonds)
                new_rw = Chem.RWMol(rw)
                new_rw.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                # Convert RWMol to SMILES
                try:
                    new_mol = new_rw.GetMol()
                    Chem.SanitizeMol(new_mol)
                    return Chem.MolToSmiles(new_mol)
                except:
                    return None
        
        return None
    
    def validate(self, smiles, max_atoms=30):
        """Validate molecule against configured atom set with single/double bonds only.

        Validation rules:
            - Single connected component (no fragments)
            - No triple bonds
            - Only atoms from the configured atom set
            - Atom count within [5, max_atoms]
            - Molecular weight within allowed range
            - No iodine (even in drug mode)
        """
        try:
            if '.' in smiles:  # Disconnected structures
                return False
            if '#' in smiles:  # Triple bonds
                return False

            # NLO mode: reject heteroatoms not in C/N/O
            if self.atom_set == 'nlo':
                if 'S' in smiles or 's' in smiles:
                    return False
                if 'F' in smiles or 'Cl' in smiles or 'Br' in smiles or 'I' in smiles:
                    return False

            # Always reject iodine
            if 'I' in smiles and 'Cl' not in smiles:
                # 'I' could appear alone (iodine) — but watch out for 'Cl' containing 'l'
                # More robust: parse the molecule and check atomic numbers below
                pass

            # Parse without sanitization first to avoid warnings
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if not mol:
                return False

            # Try to sanitize - this will fail for chemically invalid molecules
            try:
                Chem.SanitizeMol(mol)
            except:
                return False

            # Check for disconnected fragments (even if no '.' in SMILES)
            if len(Chem.GetMolFrags(mol)) > 1:
                return False

            # Check atoms against configured atom set and reject formal charges
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() not in self.allowed_atomic_numbers:
                    return False
                if atom.GetFormalCharge() != 0:
                    return False

            # Check bonds - NO TRIPLE bonds
            for bond in mol.GetBonds():
                if bond.GetBondType() == Chem.BondType.TRIPLE:
                    return False

            # Check size
            if mol.GetNumAtoms() > max_atoms:
                return False

            # Check for reasonable chemistry
            if mol.GetNumAtoms() < 5:  # Too small
                return False

            # Check molecular weight (drug mode allows larger molecules)
            mw = Descriptors.MolWt(mol)
            if self.atom_set == 'drug':
                if mw > 800 or mw < 50:
                    return False
            else:
                if mw > 600 or mw < 50:
                    return False

            return True
        except Exception:
            return False

def mutate_smiles(smiles, mutation_type, interrupted=False, max_attempts=100, atom_set='nlo'):
    mutator = MoleculeMutator(atom_set=atom_set)
    for attempt in range(max_attempts):
        if interrupted:
            break
        try:
            new_smiles = mutator.mutate(smiles, mutation_type)
            if new_smiles and mutator.validate(new_smiles):
                return new_smiles
        except Exception:
            # Continue trying if a mutation fails
            continue
    # If no valid mutation found, return original smiles
    return smiles
