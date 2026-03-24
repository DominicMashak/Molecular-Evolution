import random
import selfies as sf
from molev_utils.molecule_ops import MoleculeMutator


class SELFIESMutator:
    """Token-level mutations on SELFIES strings.

    Every mutation produces a syntactically valid SELFIES string (guaranteed by
    the SELFIES grammar), so invalid-molecule failures are eliminated at the
    encoding level.  Chemical constraints (atom set, size, weight) are still
    enforced via decoding to SMILES and passing through MoleculeMutator.validate().

    Mutation types:
        1 - substitute a random token with a different one from the alphabet
        2 - insert a random token at a random position
        3 - delete a random token

    Args:
        atom_set: Forwarded to MoleculeMutator for validation ('nlo' or 'drug').
    """

    # Tokens restricted to drug-like / NLO-relevant elements only, so the
    # initial SELFIES alphabet is filtered to atoms the project supports.
    _FULL_ALPHABET = list(sf.get_semantic_robust_alphabet())

    # Element symbols in each atom set (lower + upper because SELFIES tokens
    # look like [C], [N], [O], [Branch1], [Ring1], etc.)
    _ALLOWED_ELEMENTS = {
        'nlo': {'C', 'N', 'O'},
        'drug': {'C', 'N', 'O', 'S', 'F', 'Cl', 'Br'},
    }

    def __init__(self, atom_set: str = 'nlo'):
        if atom_set not in self._ALLOWED_ELEMENTS:
            raise ValueError(f"Unknown atom_set '{atom_set}'. Must be one of: {list(self._ALLOWED_ELEMENTS)}")
        self.atom_set = atom_set
        self._smiles_validator = MoleculeMutator(atom_set=atom_set)
        # Build a filtered token alphabet: keep structural tokens (Branch, Ring,
        # etc.) and only element tokens whose element is in the allowed set.
        self.alphabet = self._build_alphabet(atom_set)

    def _build_alphabet(self, atom_set: str) -> list:
        allowed_elements = self._ALLOWED_ELEMENTS[atom_set]
        filtered = []
        for token in self._FULL_ALPHABET:
            # Strip brackets: [C] -> C, [=N] -> N, [Branch1] -> Branch1
            inner = token.strip('[]')
            # Remove bond-order prefix characters (=, #, /)
            inner_clean = inner.lstrip('=#\\/')
            # Heuristic: if the cleaned inner starts with an uppercase letter
            # followed by optional lowercase (element symbol), check membership.
            # Structural tokens (Branch, Ring, etc.) are always kept.
            if inner_clean and inner_clean[0].isupper():
                # Extract element symbol (1 or 2 chars)
                sym = inner_clean[:2] if len(inner_clean) > 1 and inner_clean[1].islower() else inner_clean[:1]
                if sym in allowed_elements or not sym.isalpha():
                    filtered.append(token)
                elif inner_clean.startswith(('Branch', 'Ring', 'Expl')):
                    filtered.append(token)
            else:
                filtered.append(token)
        return filtered if filtered else self._FULL_ALPHABET

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def mutate(self, selfies_str: str, mutation_type: int) -> 'str | None':
        """Apply one mutation to *selfies_str* and return a new SELFIES string.

        Returns None if the mutation cannot be applied (e.g. delete on a
        single-token string).
        """
        tokens = list(sf.split_selfies(selfies_str))
        if not tokens:
            return None

        if mutation_type == 1:  # substitute
            if not tokens:
                return None
            idx = random.randrange(len(tokens))
            candidates = [t for t in self.alphabet if t != tokens[idx]]
            if not candidates:
                return None
            tokens[idx] = random.choice(candidates)

        elif mutation_type == 2:  # insert
            idx = random.randint(0, len(tokens))
            tokens.insert(idx, random.choice(self.alphabet))

        elif mutation_type == 3:  # delete
            if len(tokens) <= 1:
                return None
            idx = random.randrange(len(tokens))
            tokens.pop(idx)

        else:
            return None

        return ''.join(tokens)

    def validate(self, selfies_str: str, max_atoms: int = 30) -> bool:
        """Validate a SELFIES string by decoding to SMILES and checking constraints."""
        smiles = self.to_smiles(selfies_str)
        if smiles is None:
            return False
        return self._smiles_validator.validate(smiles, max_atoms)

    def to_smiles(self, selfies_str: str) -> 'str | None':
        """Decode SELFIES to a canonical SMILES string, or None on failure."""
        try:
            smiles = sf.decoder(selfies_str)
            return smiles if smiles else None
        except Exception:
            return None


def smiles_to_selfies(smiles: str) -> 'str | None':
    """Encode a SMILES string to SELFIES. Returns None on failure."""
    try:
        result = sf.encoder(smiles)
        return result if result else None
    except Exception:
        return None


def selfies_to_smiles(selfies_str: str) -> 'str | None':
    """Decode a SELFIES string to SMILES. Returns None on failure."""
    try:
        result = sf.decoder(selfies_str)
        return result if result else None
    except Exception:
        return None
