from rdkit import Chem
from rdkit.Chem import Descriptors
import random

class MoleculeMutator:
    """Perform mutations on molecules represented as SMILES strings.

    Supports bond-type changes, atom additions/removals, atom-type changes,
    ring additions/removals, and validation. Mutations are restricted to
    allowed atoms (C, N, O) and allowed bond types (single, double).
    """

    def __init__(self):
        """Initialize allowed atoms and bond types."""
        self.allowed_atoms = ['C', 'N', 'O']
        self.allowed_bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]
    
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
                new_atom_symbol = random.choice(self.allowed_atoms)
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
                possible_symbols = [s for s in self.allowed_atoms if s != current_symbol]
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
    
    def validate(self, smiles, max_atoms=50):
        """Validate molecule - ONLY C,N,O,H with single/double bonds, single connected component"""
        try:
            if '.' in smiles:  # Disconnected structures
                return False
            if '#' in smiles:  # Triple bonds
                return False
            if 'S' in smiles or 's' in smiles:  # Sulfur
                return False
            if 'F' in smiles or 'Cl' in smiles or 'Br' in smiles or 'I' in smiles:  # Halogens
                return False

            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return False

            # Check for disconnected fragments (even if no '.' in SMILES)
            if len(Chem.GetMolFrags(mol)) > 1:
                return False

            # Check atoms - ONLY C, N, O, H allowed
            allowed_atoms = {1, 6, 7, 8}  # H, C, N, O
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() not in allowed_atoms:
                    return False

            # Check bonds - NO TRIPLE bonds
            for bond in mol.GetBonds():
                if bond.GetBondType() == Chem.BondType.TRIPLE:
                    return False

            # Check size
            if mol.GetNumAtoms() > max_atoms:
                return False

            # Check for reasonable chemistry
            if mol.GetNumAtoms() < 4:  # Too small
                return False

            # Check molecular weight
            mw = Descriptors.MolWt(mol)
            if mw > 600 or mw < 50:  # Adjusted for CNOH only
                return False

            return True
        except Exception:
            return False

def mutate_smiles(smiles, mutation_type, interrupted=False, max_attempts=100):
    mutator = MoleculeMutator()
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