from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Individual:
    """Represents an individual in the population for NSGA-II algorithm.

    Encapsulates the properties of a molecular individual, including its SMILES,
    objective values, surrogate predictions, and features for archiving/selection.
    Supports both quantum chemistry (NLO) and drug-design (SmartCADD) objectives.
    """
    smiles: str
    objectives: Optional[List[float]] = None  # [beta, natoms] or [qed, sa_score, ...]
    beta_surrogate: float = 0.0
    natoms: int = 0
    rank: float = field(default=float('inf'))
    crowding_distance: float = 0.0
    dominated_by: int = 0
    dominates: List = field(default_factory=list)
    generation: int = 0
    # Quantum chemistry properties
    homo_lumo_gap: float = 0.0
    transition_dipole: float = 0.0
    oscillator_strength: float = 0.0
    gamma: float = 0.0
    alpha_mean: float = 0.0
    # Drug-design properties (SmartCADD)
    qed: float = 0.0
    sa_score: float = 10.0
    docking_score: float = 0.0
    lipinski_violations: int = 0
    mol_weight: float = 0.0
    logp: float = 0.0
    tpsa: float = 0.0
    admet_pass: float = 0.0

    def __post_init__(self):
        if not self.objectives:
            # Build objectives from explicit fields
            self.objectives = [float(self.beta_surrogate), float(self.natoms)]
        else:
            # Sync explicit fields from provided objectives
            if len(self.objectives) > 0:
                try:
                    self.beta_surrogate = float(self.objectives[0])
                except Exception:
                    self.beta_surrogate = 0.0
            if len(self.objectives) > 1:
                try:
                    self.natoms = int(self.objectives[1])
                except Exception:
                    self.natoms = int(self.natoms or 0)
