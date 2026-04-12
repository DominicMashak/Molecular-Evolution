"""
SmartCADD Interface for Molecular Evolution

Provides a fitness evaluation interface using SmartCADD's drug discovery pipeline
as an alternative to quantum chemistry calculations. Supports molecular descriptors,
ADMET filtering, docking scores, and ML model predictions.

Usage:
    interface = SmartCADDInterface(
        smartcadd_path="/path/to/SmartCADD",
        mode="descriptors",  # or "docking" for full docking pipeline
        protein_code="1AQ1",  # required for docking mode
    )
    result = interface.calculate("c1ccccc1")
"""

import sys
import os
import logging
import re
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, RDConfig
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.Lipinski import NumAromaticRings

logger = logging.getLogger(__name__)

# SA Score setup - RDKit's synthetic accessibility score
_sa_score_module = None

def _load_sa_score():
    """Load RDKit's SA Score module."""
    global _sa_score_module
    if _sa_score_module is not None:
        return _sa_score_module
    sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
    if sa_path not in sys.path:
        sys.path.insert(0, sa_path)
    import sascorer
    _sa_score_module = sascorer
    return _sa_score_module


class SmartCADDInterface:
    """Interface to SmartCADD drug discovery evaluation pipeline.

    Mirrors the QuantumChemistryInterface API so it can be used as a drop-in
    replacement in the NSGA-II optimizer and other algorithms.

    Modes:
        - "descriptors": Fast mode using RDKit descriptors + QED + SA score +
          SmartCADD PAINS filtering. Good for benchmarking.
        - "docking": Full pipeline using SmartCADD's ADMET filter, 3D generation,
          XTB optimization, and Smina docking. Requires SmartCADD installation
          and external tools (xtb, smina, openbabel).

    The calculate() method returns a dict with the same structure as
    QuantumChemistryInterface.calculate(), plus drug-design-specific keys.
    """

    def __init__(self,
                 smartcadd_path: str = None,
                 mode: str = "descriptors",
                 protein_code: str = None,
                 protein_path: str = None,
                 alert_collection_path: str = None,
                 docking_exhaustiveness: int = 8,
                 verbose: bool = False):
        """
        Initialize SmartCADD interface.

        Args:
            smartcadd_path: Path to SmartCADD repository root. If None, attempts
                to find it automatically.
            mode: Evaluation mode - "descriptors" or "docking".
            protein_code: PDB code for docking target (required for docking mode).
            protein_path: Local path to protein PDB file (optional, overrides fetch).
            alert_collection_path: Path to ADMET alert collection CSV. If None,
                uses SmartCADD's example data.
            docking_exhaustiveness: Smina exhaustiveness parameter.
            verbose: Enable verbose output.
        """
        self.mode = mode.lower()
        self.protein_code = protein_code
        self.protein_path = protein_path
        self.docking_exhaustiveness = docking_exhaustiveness
        self.verbose = verbose

        # For compatibility with QuantumChemistryInterface API
        self.calculator_type = f"smartcadd_{mode}"

        # Find SmartCADD
        self.smartcadd_path = self._find_smartcadd(smartcadd_path)
        if not self.smartcadd_path:
            raise RuntimeError(
                "SmartCADD installation not found. Provide smartcadd_path or "
                "install SmartCADD in a standard location."
            )

        # Add SmartCADD to path so `import smartcadd` resolves
        smartcadd_str = str(self.smartcadd_path)
        if smartcadd_str not in sys.path:
            sys.path.insert(0, smartcadd_str)
        logger.info(f"SmartCADD found at: {self.smartcadd_path}")

        # Set up alert collection path
        if alert_collection_path:
            self.alert_collection_path = alert_collection_path
        else:
            default_alerts = (self.smartcadd_path / "examples" /
                              "example_data" / "alert_collection.csv")
            if not default_alerts.exists():
                raise FileNotFoundError(
                    f"Alert collection not found at {default_alerts}. "
                    "Provide alert_collection_path explicitly."
                )
            self.alert_collection_path = str(default_alerts)

        # Initialize SmartCADD ADMET/PAINS filter
        from smartcadd.filters import ADMETFilter
        self._admet_filter = ADMETFilter(
            alert_collection_path=self.alert_collection_path
        )
        logger.info("SmartCADD ADMETFilter initialized successfully")

<<<<<<< Updated upstream
        # Validate mode requirements
        if self.mode == "docking":
            if not self.protein_code and not self.protein_path:
                raise ValueError("Docking mode requires protein_code or protein_path")

        logger.info(f"SmartCADD interface initialized (mode={self.mode})")

=======
        # Validate mode requirements and auto-prepare protein
        if self.mode == "docking":
            if not self.protein_code and not self.protein_path:
                raise ValueError("Docking mode requires protein_code or protein_path")
            self._protein_dir = self._prepare_protein()
        else:
            self._protein_dir = None

        logger.info(f"SmartCADD interface initialized (mode={self.mode})")

    def _prepare_protein(self) -> Path:
        """Auto-download and prepare protein files via protein_manager."""
        import sys as _sys
        _repo_root = str(Path(__file__).parent.parent)
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)
        from drug.protein_manager import ensure_protein_ready
        protein_dir = ensure_protein_ready(
            protein_code=self.protein_code,
            protein_path=self.protein_path,
        )
        logger.info(f"Protein {self.protein_code} ready at {protein_dir}")
        return protein_dir

>>>>>>> Stashed changes
    def _find_smartcadd(self, explicit_path: str = None) -> Path:
        """Find SmartCADD installation directory."""
        if explicit_path:
            p = Path(explicit_path)
            if (p / "smartcadd").is_dir():
                return p
            raise FileNotFoundError(
                f"SmartCADD not found at {explicit_path} "
                "(expected 'smartcadd' subdirectory)"
            )

        candidates = [
            Path(__file__).parent.parent.parent / "SmartCADD",
            Path.home() / "SmartCADD",
            Path.home() / "Documents" / "GitHub" / "SmartCADD",
            Path.cwd() / ".." / "SmartCADD",
        ]

        for p in candidates:
            p = p.resolve()
            if p.is_dir() and (p / "smartcadd").is_dir():
                return p

        return None

    def calculate(self, smiles: str, charge: int = 0, spin: int = 1) -> dict:
        """
        Evaluate a molecule for drug-design properties.

        API-compatible with QuantumChemistryInterface.calculate().

        Args:
            smiles: SMILES string.
            charge: Unused (kept for API compatibility).
            spin: Unused (kept for API compatibility).

        Returns:
            Dict with drug-design properties.
        """
        result = self._default_result(smiles)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result['error'] = "Invalid SMILES"
            return result

        # Basic descriptors
        result['natoms'] = mol.GetNumAtoms()
        result['mol_weight'] = Descriptors.MolWt(mol)
        result['logp'] = Descriptors.MolLogP(mol)
        result['hbd'] = Descriptors.NumHDonors(mol)
        result['hba'] = Descriptors.NumHAcceptors(mol)
        result['tpsa'] = Descriptors.TPSA(mol)
        result['rotatable_bonds'] = CalcNumRotatableBonds(mol)
        result['aromatic_rings'] = NumAromaticRings(mol)

        # QED (Quantitative Estimate of Drug-likeness)
        result['qed'] = QED.qed(mol)

        # SA Score (Synthetic Accessibility)
        sa_module = _load_sa_score()
        result['sa_score'] = sa_module.calculateScore(mol)

        # Lipinski Rule of Five violations
        violations = 0
        if result['mol_weight'] > 500:
            violations += 1
        if result['logp'] > 5:
            violations += 1
        if result['hbd'] > 5:
            violations += 1
        if result['hba'] > 10:
            violations += 1
        result['lipinski_violations'] = violations

        # Molecular weight range distance (optimal drug range: 150-500 Da)
        mol_weight = result['mol_weight']
        if mol_weight < 150.0:
            result['mol_weight_range_distance'] = 150.0 - mol_weight
        elif mol_weight > 500.0:
            result['mol_weight_range_distance'] = mol_weight - 500.0
        else:
            result['mol_weight_range_distance'] = 0.0

        # LogP range distance (optimal: 0-5)
        logp = result['logp']
        if logp < 0.0:
            result['logp_range_distance'] = abs(logp)
        elif logp > 5.0:
            result['logp_range_distance'] = logp - 5.0
        else:
            result['logp_range_distance'] = 0.0

        # TPSA range distance (optimal: 20-130 Å² for oral bioavailability)
        tpsa = result['tpsa']
        if tpsa < 20.0:
            result['tpsa_range_distance'] = 20.0 - tpsa
        elif tpsa > 130.0:
            result['tpsa_range_distance'] = tpsa - 130.0
        else:
            result['tpsa_range_distance'] = 0.0

<<<<<<< Updated upstream
        # ADMET / PAINS filtering via SmartCADD
        result['admet_pass'] = self._evaluate_admet(smiles)

        # Docking (only in docking mode)
        if self.mode == "docking":
=======
        # ADMET / PAINS filtering via SmartCADD.
        # This is the cheap pre-filter — molecules that fail never reach docking.
        result['admet_pass'] = self._evaluate_admet(smiles)

        # Docking (only in docking mode, only for ADMET-passing molecules)
        if self.mode == "docking":
            if not result['admet_pass']:
                # ADMET/PAINS failure — skip expensive docking.
                # docking_score stays at 0.0 (worst possible: real scores are ≤ 0 kcal/mol).
                logger.debug(f"ADMET filter failed for {smiles} — skipping docking")
                return result
>>>>>>> Stashed changes
            try:
                result['docking_score'] = self._evaluate_docking(smiles)
            except RuntimeError as e:
                logger.error(f"Docking failed for {smiles}: {e}")
                result['error'] = str(e)
                return result

        if self.verbose:
            logger.info(f"SmartCADD result for {smiles}: "
                        f"QED={result['qed']:.3f}, SA={result['sa_score']:.2f}, "
                        f"Lipinski={result['lipinski_violations']}, "
                        f"ADMET={'PASS' if result['admet_pass'] else 'FAIL'}")

        return result

    def _evaluate_admet(self, smiles: str) -> float:
        """Run SmartCADD ADMET/PAINS filtering. Returns 1.0 for pass, 0.0 for fail."""
        from smartcadd.data import Compound
        compound = Compound(smiles=smiles, id="eval")
        passes = self._admet_filter._filter(compound)
        if not passes:
            logger.debug(f"SmartCADD PAINS filter failed for {smiles}")
        return 1.0 if passes else 0.0

    def _evaluate_docking(self, smiles: str) -> float:
<<<<<<< Updated upstream
        """Run Smina docking. Returns binding affinity (lower = better)."""
=======
        """Run Smina docking. Returns binding affinity (lower = better).

        SmartCADD's SminaDockingFilter resolves receptor and ligand mol2 files
        via relative paths from the CWD, so we temporarily cd to the prepared
        protein directory where those files live.
        """
>>>>>>> Stashed changes
        from smartcadd.data import Compound
        from smartcadd.modules import SMILETo3D, PDBToPDBQT
        from smartcadd.filters import SminaDockingFilter

        with tempfile.TemporaryDirectory() as tmpdir:
            compound = Compound(smiles=smiles, id="dock_eval")

            # Generate 3D coordinates
            smile_to_3d = SMILETo3D(modify=True, output_dir=tmpdir)
            result_batch = smile_to_3d.run([compound])
            if not result_batch or result_batch[0] is None:
                raise RuntimeError(f"3D coordinate generation failed for {smiles}")
            compound = result_batch[0]

            if not hasattr(compound, 'pdb_path') or compound.pdb_path is None:
                raise RuntimeError(f"No PDB path generated for {smiles}")

            # Convert to PDBQT
            pdbqt_mod = PDBToPDBQT(output_dir=tmpdir)
            result_batch = pdbqt_mod.run([compound])
            if not result_batch or result_batch[0] is None:
                raise RuntimeError(f"PDBQT conversion failed for {smiles}")
            compound = result_batch[0]

            if not hasattr(compound, 'pdbqt_path') or compound.pdbqt_path is None:
                raise RuntimeError(f"No PDBQT path generated for {smiles}")

<<<<<<< Updated upstream
            # Run docking
            docking_filter = SminaDockingFilter(
                protein_code=self.protein_code,
                optimized_pdb_dir=tmpdir,
                protein_path=self.protein_path,
                output_dir=tmpdir,
            )
            docking_filter.run([compound])
=======
            # Run docking — cd to protein dir so SmartCADD's relative paths
            # ({code}.pdbqt, {code}_lig.mol2) resolve correctly.
            # We monkey-patch _load_and_preprocess_protein to a no-op because:
            #   (a) protein_manager already prepared all files, and
            #   (b) SmartCADD would try to cmd.fetch(protein_code) from RCSB,
            #       which fails for codes like "6WPJ_A" (not a 4-char RCSB id).
            _orig_cwd = os.getcwd()
            try:
                os.chdir(str(self._protein_dir))
                docking_filter = SminaDockingFilter(
                    protein_code=self.protein_code,
                    optimized_pdb_dir=tmpdir,
                    protein_path=None,
                    output_dir=tmpdir,
                )
                docking_filter._load_and_preprocess_protein = lambda: None
                docking_filter.run([compound])
            finally:
                os.chdir(_orig_cwd)
>>>>>>> Stashed changes

            # Parse docking score from output
            docked_path = os.path.join(tmpdir, "dock_eval_docked.pdb")
            if not os.path.exists(docked_path):
                raise RuntimeError(f"Docked output file not found for {smiles}")

            with open(docked_path, 'r') as f:
                content = f.read()
            score_match = re.search(
                r'REMARK\s+minimizedAffinity\s+([-+]?\d+\.?\d*)',
                content
            )
            if not score_match:
                raise RuntimeError(
                    f"Could not parse docking score from {docked_path}"
                )
            return float(score_match.group(1))

    def _default_result(self, smiles: str) -> dict:
        """Return default result dict with all keys initialized."""
        return {
            'smiles': smiles,
            'natoms': 0,
            'mol_weight': 0.0,
            'logp': 0.0,
            'hbd': 0,
            'hba': 0,
            'tpsa': 0.0,
            'rotatable_bonds': 0,
            'aromatic_rings': 0,
            'qed': 0.0,
            'sa_score': 10.0,
            'lipinski_violations': 4,
            'admet_pass': 0.0,
            'docking_score': 0.0,
            # Keep QC-compatible keys with default values so objective
            # mapping doesn't break if mixed objectives are used
            'beta_mean': 0.0,
            'homo_lumo_gap': 0.0,
            'total_energy': 0.0,
            'alpha_mean': 0.0,
            'gamma': 0.0,
            'dipole_moment': 0.0,
            'transition_dipole': 0.0,
            'oscillator_strength': 0.0,
            'beta_gamma_ratio': 0.0,
            'total_energy_atom_ratio': 0.0,
            'alpha_range_distance': 0.0,
            'homo_lumo_gap_range_distance': 0.0,
            'error': None,
        }
