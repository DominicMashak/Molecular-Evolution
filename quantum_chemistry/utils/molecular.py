#!/usr/bin/env python3
"""
Molecular utilities for SMILES processing and geometry generation.
"""

import numpy as np
from typing import Optional, Tuple, Any
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def canonicalize_smiles(smiles: str) -> str:
    """
    Canonicalize a SMILES string using RDKit.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Canonical SMILES string
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else smiles
    except:
        return smiles


def smiles_to_geometry(smiles: str, optimize: bool = True, 
                       max_iters: int = 200) -> Tuple[Optional[np.ndarray], 
                                                       Optional[str], 
                                                       Optional[Any], 
                                                       Optional[np.ndarray]]:
    """
    Convert SMILES to 3D geometry using RDKit.
    
    Args:
        smiles: SMILES string
        optimize: Whether to optimize geometry with UFF
        max_iters: Maximum iterations for optimization
        
    Returns:
        Tuple of (coordinates, formula, rdkit_mol, atomic_numbers)
        Returns (None, None, None, None) on failure
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            # Embedding failed, try with different parameters
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.maxAttempts = 50
            result = AllChem.EmbedMolecule(mol, params)
            
            if result == -1:
                raise ValueError("Could not generate 3D coordinates")
        
        # Optimize geometry if requested
        if optimize:
            AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
        
        # Extract coordinates
        conf = mol.GetConformer()
        coords = []
        atomic_numbers = []
        
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            coords.append([pos.x, pos.y, pos.z])
            atomic_numbers.append(atom.GetAtomicNum())
        
        coords = np.array(coords)
        atomic_numbers = np.array(atomic_numbers)
        
        # Get molecular formula
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        
        return coords, formula, mol, atomic_numbers
        
    except Exception as e:
        return None, None, None, None


def get_molecular_descriptors(mol: Any) -> dict:
    """
    Calculate molecular descriptors relevant for NLO properties.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary of molecular descriptors
    """
    descriptors = {}
    
    try:
        # Basic descriptors
        descriptors['num_atoms'] = mol.GetNumAtoms()
        descriptors['num_heavy_atoms'] = Descriptors.HeavyAtomCount(mol)
        descriptors['num_bonds'] = mol.GetNumBonds()
        descriptors['molecular_weight'] = Descriptors.MolWt(mol)
        
        # Aromaticity and conjugation
        descriptors['num_aromatic_rings'] = Descriptors.NumAromaticRings(mol)
        descriptors['num_aromatic_atoms'] = len([a for a in mol.GetAtoms() if a.GetIsAromatic()])
        descriptors['num_conjugated_bonds'] = len([b for b in mol.GetBonds() if b.GetIsConjugated()])
        
        # Electronic descriptors
        descriptors['num_heteroatoms'] = Descriptors.NumHeteroatoms(mol)
        descriptors['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
        
        # Topology
        descriptors['tpsa'] = Descriptors.TPSA(mol)
        descriptors['logp'] = Descriptors.MolLogP(mol)
        
        # Check for specific functional groups
        descriptors['has_nitro'] = check_nitro_group(mol)
        descriptors['has_amino'] = check_amino_group(mol)
        descriptors['has_carbonyl'] = check_carbonyl_group(mol)
        descriptors['has_cyano'] = check_cyano_group(mol)
        
        # Check for donor-acceptor character
        descriptors['has_donor'] = check_donor_groups(mol)
        descriptors['has_acceptor'] = check_acceptor_groups(mol)
        descriptors['is_push_pull'] = descriptors['has_donor'] and descriptors['has_acceptor']
        
    except Exception as e:
        pass
    
    return descriptors


def check_nitro_group(mol: Any) -> bool:
    """Check if molecule contains nitro group."""
    nitro_smarts = '[N+](=O)[O-]'
    pattern = Chem.MolFromSmarts(nitro_smarts)
    return mol.HasSubstructMatch(pattern) if pattern else False


def check_amino_group(mol: Any) -> bool:
    """Check if molecule contains amino group."""
    amino_smarts = '[NX3;H2,H1,H0]'
    pattern = Chem.MolFromSmarts(amino_smarts)
    return mol.HasSubstructMatch(pattern) if pattern else False


def check_carbonyl_group(mol: Any) -> bool:
    """Check if molecule contains carbonyl group."""
    carbonyl_smarts = '[CX3]=[OX1]'
    pattern = Chem.MolFromSmarts(carbonyl_smarts)
    return mol.HasSubstructMatch(pattern) if pattern else False


def check_cyano_group(mol: Any) -> bool:
    """Check if molecule contains cyano group."""
    cyano_smarts = '[C]#[N]'
    pattern = Chem.MolFromSmarts(cyano_smarts)
    return mol.HasSubstructMatch(pattern) if pattern else False


def check_donor_groups(mol: Any) -> bool:
    """
    Check if molecule has electron donor groups.
    Common donors: -NH2, -NR2, -OH, -OR, -SH, -SR
    """
    donor_smarts = [
        '[NX3;H2,H1,H0]',  # Amino groups
        '[OX2H]',          # Hydroxyl
        '[OX2]C',          # Ether
        '[SX2H]',          # Thiol
        '[SX2]C',          # Thioether
    ]
    
    for smarts in donor_smarts:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            return True
    
    return False


def check_acceptor_groups(mol: Any) -> bool:
    """
    Check if molecule has electron acceptor groups.
    Common acceptors: -NO2, -CN, -CHO, -COR, -COOH, -SO2R
    """
    acceptor_smarts = [
        '[N+](=O)[O-]',      # Nitro
        '[C]#[N]',           # Cyano
        '[CX3H1](=O)',       # Aldehyde
        '[CX3](=O)[#6]',     # Ketone
        '[CX3](=O)[OX2H1]',  # Carboxylic acid
        '[SX4](=O)(=O)',     # Sulfonyl
    ]
    
    for smarts in acceptor_smarts:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            return True
    
    return False


def estimate_conjugation_length(mol: Any) -> int:
    """
    Estimate the effective conjugation length in the molecule.

    Args:
        mol: RDKit molecule object

    Returns:
        Estimated conjugation length (number of conjugated atoms)
    """
    try:
        # Find all conjugated bonds
        conjugated_bonds = [b for b in mol.GetBonds() if b.GetIsConjugated()]

        if not conjugated_bonds:
            return 0

        # Build graph of conjugated atoms (undirected adjacency list)
        conjugated_atoms = set()
        adj = {}
        for bond in conjugated_bonds:
            a = bond.GetBeginAtomIdx()
            bidx = bond.GetEndAtomIdx()
            conjugated_atoms.add(a)
            conjugated_atoms.add(bidx)
            adj.setdefault(a, set()).add(bidx)
            adj.setdefault(bidx, set()).add(a)

        # Safety cutoff: avoid exponential search on huge graphs
        MAX_ATOMS_FOR_EXHAUSTIVE = 50
        if len(conjugated_atoms) > MAX_ATOMS_FOR_EXHAUSTIVE:
            # fallback: return number of unique conjugated atoms as conservative estimate
            return len(conjugated_atoms)

        # Depth-first search to find longest simple path starting from a node
        visited_global = set()
        longest = 0

        def dfs(node, visited):
            nonlocal longest
            longest = max(longest, len(visited))
            for nbr in adj.get(node, ()):
                if nbr not in visited:
                    visited.add(nbr)
                    dfs(nbr, visited)
                    visited.remove(nbr)

        # Run DFS from each conjugated atom
        for start in conjugated_atoms:
            dfs(start, {start})

        return int(longest)

    except Exception:
        # On any failure, fall back to conservative estimate
        try:
            conjugated_bonds = [b for b in mol.GetBonds() if b.GetIsConjugated()]
            conjugated_atoms = set()
            for bond in conjugated_bonds:
                conjugated_atoms.add(bond.GetBeginAtomIdx())
                conjugated_atoms.add(bond.GetEndAtomIdx())
            return len(conjugated_atoms)
        except Exception:
            return 0


def get_charge_distribution(mol: Any) -> dict:
    """
    Get information about charge distribution in the molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary with charge information
    """
    charge_info = {
        'total_charge': 0,
        'num_positive': 0,
        'num_negative': 0,
        'charge_separation': 0
    }
    
    try:
        positive_positions = []
        negative_positions = []
        
        for atom in mol.GetAtoms():
            charge = atom.GetFormalCharge()
            charge_info['total_charge'] += charge
            
            if charge > 0:
                charge_info['num_positive'] += 1
                conf = mol.GetConformer()
                pos = conf.GetAtomPosition(atom.GetIdx())
                positive_positions.append([pos.x, pos.y, pos.z])
            elif charge < 0:
                charge_info['num_negative'] += 1
                conf = mol.GetConformer()
                pos = conf.GetAtomPosition(atom.GetIdx())
                negative_positions.append([pos.x, pos.y, pos.z])
        
        # Calculate charge separation distance
        if positive_positions and negative_positions:
            pos_center = np.mean(positive_positions, axis=0)
            neg_center = np.mean(negative_positions, axis=0)
            charge_info['charge_separation'] = np.linalg.norm(pos_center - neg_center)
            
    except:
        pass
    
    return charge_info