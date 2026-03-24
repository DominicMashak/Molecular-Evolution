import os
import numpy as np
from typing import List, Tuple, Optional, Dict

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
Molecular Embedder using ChemBERTa-2 MTR for CVT-MOME behavior descriptors.
Embeds SMILES strings into fixed-length vectors via transformer + UMAP.
"""


class MolecularEmbedder:
    """
    Embeds molecules (SMILES) into low-dimensional vectors using a pretrained
    ChemBERTa-2 transformer model with UMAP dimensionality reduction.

    The embedding pipeline:
    1. Tokenize SMILES with the model's tokenizer
    2. Extract 768D hidden states from the transformer (mean pooling)
    3. Reduce to N dimensions via UMAP (fitted on an initialization sample)

    UMAP preserves local neighborhood structure in chemical space, so
    chemically similar molecules cluster in nearby Voronoi cells in CVT-MOME.
    """

    def __init__(self, model_name: str = 'DeepChem/ChemBERTa-77M-MTR',
                 n_components: int = 8,
                 device: str = 'auto',
                 random_state: int = 42):
        """
        Args:
            model_name: HuggingFace model identifier.
            n_components: Number of UMAP output dimensions.
            device: Device for transformer inference. Options:
                    'auto' - auto-detect best available (cuda > mps > cpu)
                    'cuda' - NVIDIA/AMD ROCm GPU
                    'mps'  - Apple Metal (M-series Macs)
                    'cpu'  - CPU fallback
            random_state: Seed for UMAP fitting.
        """
        self.model_name = model_name
        self.n_components = n_components
        self.random_state = random_state

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

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}.")

        self.reducer: Optional[UMAP] = None
        self.measure_bounds: Optional[List[Tuple[float, float]]] = None
        self._cache: Dict[str, np.ndarray] = {}

    def _embed_raw(self, smiles_list: List[str]) -> np.ndarray:
        """
        Embed a list of SMILES into raw 768D vectors (before UMAP).

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Array of shape (len(smiles_list), hidden_dim).
        """
        embeddings = []
        batch_size = 64

        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i:i + batch_size]
                tokens = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=512, return_tensors='pt'
                ).to(self.device)

                outputs = self.model(**tokens)
                # Mean pooling over token positions (excluding padding)
                attention_mask = tokens['attention_mask'].unsqueeze(-1)
                hidden = outputs.last_hidden_state * attention_mask
                pooled = hidden.sum(dim=1) / attention_mask.sum(dim=1)

                embeddings.append(pooled.cpu().numpy())

        return np.vstack(embeddings)

    def fit(self, smiles_list: List[str]):
        """
        Fit UMAP on a sample of SMILES to establish the embedding-to-measure
        transformation and estimate measure bounds for CVT.

        This learns a manifold from ChemBERTa-2's 768D space to an N-dimensional
        UMAP space by fitting on a representative sample of molecules. The fitted
        UMAP model is then used to transform all future molecules consistently.

        Args:
            smiles_list: Sample SMILES for UMAP fitting. Typically 500-2000 for
                         small projects, up to 5000-10000 for production. More
                         samples = better manifold coverage but slower fitting.
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
