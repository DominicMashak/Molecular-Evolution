import os
import re
import numpy as np
from typing import List, Tuple, Optional, Dict

# ---------------------------------------------------------------------------
# Workaround: some environments have TensorFlow installed with __spec__ = None.
# torch._dynamo.trace_rules iterates sys.modules via importlib.util.find_spec
# at import time; a None __spec__ raises ValueError and crashes the import.
# Patch find_spec to return None instead so dynamo can skip broken modules.
# This must run before 'import torch' to intercept the first dynamo load.
# ---------------------------------------------------------------------------
import importlib.util as _iutil

_orig_find_spec = _iutil.find_spec


def _safe_find_spec(name, package=None, path=None, _orig=_iutil.find_spec):
    try:
        return _orig(name, package)
    except (ValueError, AttributeError):
        return None


_iutil.find_spec = _safe_find_spec
del _safe_find_spec, _iutil
# ---------------------------------------------------------------------------

# Enable AMD ROCm GPU support in WSL2 (use system HIP library)
if os.path.exists('/opt/rocm/lib/libamdhip64.so'):
    import ctypes
    try:
        ctypes.CDLL('/opt/rocm/lib/libamdhip64.so', mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
    os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
from transformers import AutoTokenizer, AutoModel
from umap import UMAP

"""
Molecular Embedder for CVT-MOME / CMA-MAE behavior descriptors.

Supports three models:
  - ChemBERTa-2 MTR ('DeepChem/ChemBERTa-77M-MTR'): for SMILES / SELFIES genotypes
  - PolyBERT ('kuelumbus/polyBERT'): for BigSMILES polymer genotypes
    (Kuenneth et al. 2023, Nature Comms — trained on ~100M polymer SMILES)
  - MatText-slices-2m ('n0w0f/MatText-slices-2m'): for SLICES crystal genotypes
    (BERT-style, 33M params, trained on 2M crystal structures from NOMAD,
     uses bert-base-uncased tokenizer since no custom tokenizer is shipped)

Pass input_format='bigsmiles' when using PolyBERT (extracts repeat-unit SMILES).
Pass input_format='slices' when using MatText (SLICES strings passed directly).
"""

# Regex for BigSMILES repeat unit extraction (matches first [<]...[>])
_BS_RU_RE = re.compile(r'\[<\](.*?)\[>\]')

# Models whose tokenizer is not bundled in the HuggingFace repo —
# map model name prefix → tokenizer to load instead.
_EXTERNAL_TOKENIZERS = {
    'n0w0f/MatText': 'bert-base-uncased',
}


class MolecularEmbedder:
    """
    Embeds molecules / crystals into low-dimensional vectors using a pretrained
    transformer model with UMAP dimensionality reduction.

    Supports:
    - ChemBERTa-2 MTR (default): SMILES / SELFIES genotypes
    - PolyBERT ('kuelumbus/polyBERT'): BigSMILES polymer genotypes
    - MatText-slices-2m ('n0w0f/MatText-slices-2m'): SLICES crystal genotypes

    The embedding pipeline:
    1. Optionally preprocess input strings (BigSMILES → repeat unit SMILES)
    2. Tokenize with the model's tokenizer
    3. Extract hidden states via mean pooling over token positions
    4. Reduce to N dimensions via UMAP (fitted on an initialization sample)

    UMAP preserves local neighborhood structure so chemically / structurally
    similar structures cluster in nearby Voronoi cells in CVT archives.
    """

    def __init__(self, model_name: str = 'DeepChem/ChemBERTa-77M-MTR',
                 n_components: int = 8,
                 device: str = 'auto',
                 random_state: int = 42,
                 input_format: str = 'smiles',
                 tokenizer_name: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace model identifier.
                        'DeepChem/ChemBERTa-77M-MTR' for SMILES/SELFIES (default).
                        'kuelumbus/polyBERT' for BigSMILES polymer genotypes.
                        'n0w0f/MatText-slices-2m' for SLICES crystal genotypes.
            n_components: Number of UMAP output dimensions.
            device: Device for transformer inference. Options:
                    'auto' - auto-detect best available (cuda > mps > cpu)
                    'cuda' - NVIDIA/AMD ROCm GPU
                    'mps'  - Apple Metal (M-series Macs)
                    'cpu'  - CPU fallback
            random_state: Seed for UMAP fitting.
            input_format: 'smiles' (default) passes strings directly to the tokenizer.
                          'bigsmiles' extracts the first repeat unit SMILES from a
                          canonical BigSMILES string before tokenization.
                          'slices' passes SLICES crystal strings directly (no preprocessing).
            tokenizer_name: Optional override for the tokenizer HuggingFace path.
                            Needed when the model repo does not bundle a tokenizer
                            (e.g. MatText-slices-2m uses bert-base-uncased).
                            If None, auto-detected from _EXTERNAL_TOKENIZERS or defaults
                            to model_name.
        """
        self.model_name = model_name
        self.n_components = n_components
        self.random_state = random_state
        if input_format not in ('smiles', 'bigsmiles', 'slices'):
            raise ValueError(
                f"input_format must be 'smiles', 'bigsmiles', or 'slices', got '{input_format}'")
        self.input_format = input_format

        # Resolve device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        # Resolve tokenizer: some models (MatText) don't ship a tokenizer in their repo
        if tokenizer_name:
            _tok_name = tokenizer_name
        else:
            _tok_name = next(
                (v for k, v in _EXTERNAL_TOKENIZERS.items() if model_name.startswith(k)),
                model_name,
            )

        print(f"Loading {model_name} (tokenizer: {_tok_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(_tok_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}.")

        self.reducer: Optional[UMAP] = None
        self.measure_bounds: Optional[List[Tuple[float, float]]] = None
        self._cache: Dict[str, np.ndarray] = {}

    def _preprocess(self, s: str) -> str:
        """Preprocess an input string before tokenization.

        For input_format='bigsmiles', extracts the first repeat unit SMILES
        from a canonical BigSMILES string so PolyBERT sees the polymer backbone.
        Falls back to the raw string if extraction fails.

        For input_format='slices', passes the SLICES crystal string through unchanged.
        """
        if self.input_format == 'bigsmiles':
            m = _BS_RU_RE.search(s)
            return m.group(1) if m else s
        # 'smiles' and 'slices': pass through unchanged
        return s

    def _embed_raw(self, smiles_list: List[str]) -> np.ndarray:
        """
        Embed a list of strings into raw hidden-state vectors (before UMAP).

        Applies _preprocess() to each string (a no-op for input_format='smiles').

        Args:
            smiles_list: List of SMILES or BigSMILES strings.

        Returns:
            Array of shape (len(smiles_list), hidden_dim).
        """
        embeddings = []
        batch_size = 64

        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                batch = [self._preprocess(s) for s in smiles_list[i:i + batch_size]]
                n_real = len(batch)
                # Pad to fixed batch_size so every GPU call has shape (64, 128).
                # Constant tensor shape → one ROCm kernel compiled at warmup, never again.
                if n_real < batch_size:
                    batch = batch + [batch[0]] * (batch_size - n_real)

                tokens = self.tokenizer(
                    batch, padding='max_length', truncation=True,
                    max_length=128, return_tensors='pt'
                ).to(self.device)

                outputs = self.model(**tokens)
                # Mean pooling over token positions (excluding padding)
                attention_mask = tokens['attention_mask'].unsqueeze(-1)
                hidden = outputs.last_hidden_state * attention_mask
                pooled = hidden.sum(dim=1) / attention_mask.sum(dim=1)

                embeddings.append(pooled[:n_real].cpu().numpy())

        return np.vstack(embeddings)

    def fit(self, smiles_list: List[str]):
        """
        Fit UMAP on a sample of SMILES to establish the embedding-to-measure
        transformation and estimate measure bounds for CVT.

        This learns a manifold from the transformer's hidden space to an N-dimensional
        UMAP space by fitting on a representative sample of genotype strings. The fitted
        UMAP model is then used to transform all future strings consistently.

        Args:
            smiles_list: Sample genotype strings (SMILES, BigSMILES, or SLICES) for
                         UMAP fitting. Typically 500-2000 for small projects, up to
                         5000-10000 for production. More samples = better manifold
                         coverage but slower fitting.
        """
        print(f"Fitting UMAP on {len(smiles_list)} molecules "
              f"({self.n_components} components)...")

        raw = self._embed_raw(smiles_list)
        self.reducer = UMAP(
            n_components=self.n_components,
            n_neighbors=30,
            min_dist=0.1,
            metric='cosine',
            random_state=self.random_state,
        )
        transformed = self.reducer.fit_transform(raw)

        # Estimate bounds with margin
        mins = transformed.min(axis=0)
        maxs = transformed.max(axis=0)
        ranges = maxs - mins
        margin = 0.5 * ranges
        self.measure_bounds = [
            (float(lo - m), float(hi + m))
            for lo, hi, m in zip(mins, maxs, margin)
        ]

        print(f"UMAP fit complete.")
        print(f"Measure bounds: {[(f'{lo:.2f}', f'{hi:.2f}') for lo, hi in self.measure_bounds]}")

        # Pre-cache the fitting sample and store fitted embeddings for CVT seeding
        self._fitted_embeddings = transformed.copy()
        for smi, vec in zip(smiles_list, transformed):
            self._cache[smi] = vec

    def get_fitted_embeddings(self) -> np.ndarray:
        """
        Return the UMAP-transformed embeddings of the fitting sample.

        These are used to seed CVT centroid generation with real molecular
        embeddings rather than uniform random samples, ensuring centroids
        are placed where molecules actually live in the embedding manifold.

        Returns:
            Array of shape (n_fitting_samples, n_components).
        """
        if not hasattr(self, '_fitted_embeddings') or self._fitted_embeddings is None:
            raise RuntimeError("UMAP not fitted. Call fit() first.")
        return self._fitted_embeddings

    # DEPRECATED: Legacy alias from when this used PCA instead of UMAP.
    # Use fit() directly. Kept for backward compatibility with old scripts.
    fit_pca = fit

    def embed(self, smiles: str) -> np.ndarray:
        """
        Embed a single SMILES into an N-dim UMAP-transformed vector.
        Uses cache to avoid redundant transformer calls.

        Args:
            smiles: SMILES string.

        Returns:
            1D array of shape (n_components,).
        """
        if smiles in self._cache:
            return self._cache[smiles]

        if self.reducer is None:
            raise RuntimeError("UMAP not fitted. Call fit() first.")

        raw = self._embed_raw([smiles])
        transformed = self.reducer.transform(raw)[0]
        self._cache[smiles] = transformed
        return transformed

    def embed_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        Embed a batch of SMILES, using cache where available.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Array of shape (len(smiles_list), n_components).
        """
        if self.reducer is None:
            raise RuntimeError("UMAP not fitted. Call fit() first.")

        results = np.empty((len(smiles_list), self.n_components))
        to_compute = []
        to_compute_idx = []

        for i, smi in enumerate(smiles_list):
            if smi in self._cache:
                results[i] = self._cache[smi]
            else:
                to_compute.append(smi)
                to_compute_idx.append(i)

        if to_compute:
            raw = self._embed_raw(to_compute)
            transformed = self.reducer.transform(raw)
            for j, idx in enumerate(to_compute_idx):
                results[idx] = transformed[j]
                self._cache[to_compute[j]] = transformed[j]

        return results

    def get_measure_keys(self) -> List[str]:
        """Return property dict keys for embedding dimensions."""
        return [f'emb_{i}' for i in range(self.n_components)]

    def get_measure_bounds(self) -> List[Tuple[float, float]]:
        """Return estimated (min, max) per UMAP dimension."""
        if self.measure_bounds is None:
            raise RuntimeError("UMAP not fitted. Call fit() first.")
        return self.measure_bounds
