"""
Protein preparation manager for docking experiments.

Automatically downloads and prepares protein structures from RCSB PDB,
caching prepared files in drug/proteins/{code}/ so each protein is only
prepared once.

Supported code formats:
  - "6WPJ"   → downloads 6WPJ, uses all chains
  - "6WPJ_A" → downloads 6WPJ, extracts chain A only

Prepared files per code:
  {code}.pdb        raw downloaded / chain-extracted PDB
  {code}_clean.pdb  PDBFixer-repaired PDB (added Hs, fixed missing atoms)
  {code}_lig.mol2   co-crystallised ligand (used by SmartCADD for box centre)
  {code}.pdbqt      receptor ready for smina

Usage:
  from drug.protein_manager import ensure_protein_ready

  protein_dir = ensure_protein_ready("6WPJ_A")
  # protein_dir / "6WPJ_A.pdbqt" and "6WPJ_A_lig.mol2" now exist
"""

import os
import re
import subprocess
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

PROTEINS_DIR = Path(__file__).parent / "proteins"
RCSB_PDB_URL = "https://files.rcsb.org/download/{code}.pdb"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_protein_ready(
    protein_code: str,
    protein_path: Optional[str] = None,
    addHs_pH: float = 7.4,
    force: bool = False,
) -> Path:
    """
    Ensure {code}.pdbqt and {code}_lig.mol2 exist in drug/proteins/{code}/.

    If they are already present (and force=False) this is a no-op and returns
    the directory immediately — no network access, no processing.

    Args:
        protein_code: Code like "6WPJ" or "6WPJ_A" (chain suffix optional).
        protein_path:  Path to a local PDB file to use instead of downloading.
                       Skips RCSB fetch but still runs the full prep pipeline.
        addHs_pH:     pH used by PDBFixer when adding missing hydrogens.
        force:        Re-run preparation even if output files already exist.

    Returns:
        Path to the directory containing the prepared files.
    """
    protein_dir = PROTEINS_DIR / protein_code
    pdbqt_path = protein_dir / f"{protein_code}.pdbqt"
    lig_path = protein_dir / f"{protein_code}_lig.mol2"

    if pdbqt_path.exists() and lig_path.exists() and not force:
        logger.info(f"Protein {protein_code} already prepared at {protein_dir}")
        return protein_dir

    protein_dir.mkdir(parents=True, exist_ok=True)
    base_code, chain = _parse_protein_code(protein_code)

    # ── Step 1: obtain source PDB ──────────────────────────────────────────
    if protein_path:
        source_pdb = Path(protein_path)
        if not source_pdb.exists():
            raise FileNotFoundError(f"Provided protein_path not found: {protein_path}")
        logger.info(f"Using provided PDB: {source_pdb}")
    else:
        source_pdb = _download_pdb(base_code, protein_dir)

    # ── Step 2: extract chain + ligand, clean, convert ─────────────────────
    _prepare_protein(
        protein_code=protein_code,
        source_pdb=source_pdb,
        chain=chain,
        protein_dir=protein_dir,
        addHs_pH=addHs_pH,
    )

    if not pdbqt_path.exists():
        raise RuntimeError(
            f"Protein prep completed but {pdbqt_path} was not created. "
            "Check obabel is on PATH and the PDB has a protein chain."
        )
    if not lig_path.exists():
        raise RuntimeError(
            f"Protein prep completed but {lig_path} was not created. "
            "The structure may have no organic ligand — add one manually or "
            "provide a reference ligand mol2 file at that path."
        )

    logger.info(f"Protein {protein_code} ready in {protein_dir}")
    return protein_dir


def list_prepared_proteins() -> list:
    """Return a list of protein codes that have already been prepared."""
    if not PROTEINS_DIR.exists():
        return []
    return sorted(
        d.name for d in PROTEINS_DIR.iterdir()
        if d.is_dir()
        and (d / f"{d.name}.pdbqt").exists()
        and (d / f"{d.name}_lig.mol2").exists()
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_protein_code(code: str) -> Tuple[str, Optional[str]]:
    """
    Split "6WPJ_A" → ("6WPJ", "A"), or "1AQ1" → ("1AQ1", None).
    Handles codes like "6WPJ_AB" (multi-chain selection, passed to PyMOL as-is).
    """
    m = re.match(r'^([A-Za-z0-9]{4})_([A-Za-z0-9]+)$', code)
    if m:
        return m.group(1), m.group(2)
    return code, None


def _download_pdb(base_code: str, protein_dir: Path) -> Path:
    """Download {base_code}.pdb from RCSB into protein_dir. Returns path."""
    dest = protein_dir / f"{base_code}.pdb"
    if dest.exists():
        logger.info(f"PDB already downloaded: {dest}")
        return dest

    url = RCSB_PDB_URL.format(code=base_code.upper())
    logger.info(f"Downloading {base_code} from RCSB: {url}")
    print(f"[protein_manager] Downloading {base_code} from RCSB...")

    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to download {base_code} from RCSB "
            f"(HTTP {resp.status_code}). "
            f"Check the code is correct: {url}"
        )
    dest.write_bytes(resp.content)
    print(f"[protein_manager] Saved {dest} ({len(resp.content) / 1024:.1f} KB)")
    return dest


def _prepare_protein(
    protein_code: str,
    source_pdb: Path,
    chain: Optional[str],
    protein_dir: Path,
    addHs_pH: float = 7.4,
) -> None:
    """
    Full protein preparation pipeline:
      1. Load in PyMOL, select chain(s) (all protein if no chain suffix),
         save protein PDB + co-crystallised ligand mol2
      2. PDBFixer: add missing residues/atoms/Hs, remove water/heterogens
      3. obabel: convert cleaned PDB → PDBQT for smina
    All outputs written to protein_dir/.
    """
    from pymol import cmd

    raw_pdb = protein_dir / f"{protein_code}.pdb"
    clean_pdb = protein_dir / f"{protein_code}_clean.pdb"
    lig_mol2 = protein_dir / f"{protein_code}_lig.mol2"
    pdbqt = protein_dir / f"{protein_code}.pdbqt"

    # ── PyMOL: extract chain(s) + ligand ──────────────────────────────────
    print(f"[protein_manager] Extracting chain/ligand with PyMOL...")
    cmd.delete("all")
    cmd.load(str(source_pdb))

    # No chain suffix → use all polymer protein chains
    if chain:
        chain_sel = f" and chain {chain}"
    else:
        chain_sel = ""

    prot_sel = f"polymer.protein{chain_sel}"

    if chain:
        n_prot = cmd.count_atoms(prot_sel)
        if n_prot == 0:
            raise RuntimeError(
                f"Chain {chain} has no protein atoms in {source_pdb}. "
                f"Available chains: {_list_chains(cmd)}"
            )

    cmd.save(str(raw_pdb), selection=prot_sel, format="pdb")

    # Ligand: all organic small molecules (co-crystallised drug/fragment)
    # — not nucleic acid chains, just genuine organic heterogens
    lig_sel = f"organic{chain_sel}"
    n_lig = cmd.count_atoms(lig_sel)
    if n_lig == 0 and chain:
        # Fall back to all organic in case ligand is on a different chain
        lig_sel = "organic"
        n_lig = cmd.count_atoms(lig_sel)
    if n_lig > 0:
        cmd.save(str(lig_mol2), selection=lig_sel, format="mol2")
    else:
        logger.warning(
            f"No organic ligand found in {source_pdb}. "
            "Docking box will not be auto-defined — "
            f"provide {lig_mol2} manually."
        )
    cmd.delete("all")

    # ── PDBFixer: repair structure ─────────────────────────────────────────
    print(f"[protein_manager] Running PDBFixer...")
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    fixer = PDBFixer(filename=str(raw_pdb))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(addHs_pH)
    with open(str(clean_pdb), "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    # ── obabel: PDB → PDBQT ───────────────────────────────────────────────
    print(f"[protein_manager] Converting to PDBQT with obabel...")
    result = subprocess.run(
        ["obabel", "-ipdb", str(clean_pdb), "-opdbqt", "-O", str(pdbqt)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not pdbqt.exists():
        raise RuntimeError(
            f"obabel failed to convert {clean_pdb} → {pdbqt}:\n{result.stderr}"
        )

    # ── Add Hs to ligand mol2 (matches SmartCADD's preprocess) ────────────
    if lig_mol2.exists():
        try:
            import openbabel.pybel as pybel
            mols = list(pybel.readfile("mol2", str(lig_mol2)))
            if mols:
                mol = mols[0]
                mol.addh()
                with pybel.Outputfile("mol2", str(lig_mol2), overwrite=True) as out:
                    out.write(mol)
        except Exception as e:
            logger.warning(f"Could not add Hs to ligand mol2: {e}")

    print(f"[protein_manager] Done: {pdbqt.name}, {lig_mol2.name if lig_mol2.exists() else 'no ligand'}")


def _list_chains(cmd) -> str:
    chains = []
    cmd.iterate("polymer.protein and name CA", "chains.append(chain)",
                space={"chains": chains})
    return ", ".join(sorted(set(chains))) or "none found"
