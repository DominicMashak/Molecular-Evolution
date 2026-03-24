import os
import numpy as np
from typing import List, Optional

# Enable AMD ROCm GPU support in WSL2 (same pattern as molecular_embedder.py)
if os.path.exists('/opt/rocm/lib/libamdhip64.so'):
    import ctypes
    try:
        ctypes.CDLL('/opt/rocm/lib/libamdhip64.so', mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
    os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def _smiles_to_selfies(smiles: str) -> Optional[str]:
    try:
        import selfies as sf
        return sf.encoder(smiles)
    except Exception:
        return None


def _selfies_to_smiles(selfies_str: str) -> Optional[str]:
    try:
        import selfies as sf
        from rdkit import Chem
        smi = sf.decoder(selfies_str)
        if smi and Chem.MolFromSmiles(smi):
            return smi
        return None
    except Exception:
        return None


class SELFIESTokenizer:
    """Maps SELFIES token strings to integer indices for VAE input/output."""

    PAD = '__PAD__'  # Padding (not a real SELFIES token)
    SOS = '__SOS__'  # Start-of-sequence
    EOS = '__EOS__'  # End-of-sequence

    def __init__(self):
        self.token2idx: dict = {}
        self.idx2token: dict = {}
        self.vocab_size: int = 0

    def fit(self, selfies_list: List[str]):
        """Build vocabulary from a list of SELFIES strings."""
        import selfies as sf
        tokens: set = set()
        for sel in selfies_list:
            for tok in sf.split_selfies(sel):
                tokens.add(tok)
        all_tokens = [self.PAD, self.SOS, self.EOS] + sorted(tokens)
        self.token2idx = {t: i for i, t in enumerate(all_tokens)}
        self.idx2token = {i: t for t, i in self.token2idx.items()}
        self.vocab_size = len(all_tokens)

    def encode(self, selfies_str: str, max_len: Optional[int] = None) -> List[int]:
        """SELFIES string → padded list of token indices (SOS + tokens + EOS [+ PAD])."""
        import selfies as sf
        tokens = [self.SOS] + list(sf.split_selfies(selfies_str)) + [self.EOS]
        indices = [self.token2idx.get(t, self.pad_idx) for t in tokens]
        if max_len is not None:
            indices = indices[:max_len]
            indices += [self.pad_idx] * max(0, max_len - len(indices))
        return indices

    def decode(self, indices: List[int]) -> str:
        """Token index list → SELFIES string (stops at EOS, skips PAD/SOS)."""
        tokens = []
        for idx in indices:
            tok = self.idx2token.get(idx, self.PAD)
            if tok == self.EOS:
                break
            if tok not in (self.PAD, self.SOS):
                tokens.append(tok)
        return ''.join(tokens)

    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.PAD]

    @property
    def sos_idx(self) -> int:
        return self.token2idx[self.SOS]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[self.EOS]

    def to_dict(self) -> dict:
        return {'token2idx': self.token2idx, 'vocab_size': self.vocab_size}

    @classmethod
    def from_dict(cls, d: dict) -> 'SELFIESTokenizer':
        obj = cls()
        obj.token2idx = d['token2idx']
        obj.idx2token = {int(i): t for t, i in obj.token2idx.items()}
        obj.vocab_size = d['vocab_size']
        return obj


class _VAEModel(nn.Module):
    """GRU-based VAE encoder-decoder for SELFIES token sequences."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 latent_dim: int, pad_idx: int):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.encoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder (input = token embedding concatenated with z)
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(embed_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x: torch.Tensor):
        """x: (batch, seq_len) → (mean, logvar) each shape (batch, latent_dim)"""
        emb = self.embed(x)
        _, h = self.encoder_gru(emb)  # h: (1, batch, hidden)
        h = h.squeeze(0)              # (batch, hidden)
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(std)
        return mean

    def decode_tf(self, z: torch.Tensor, x_dec: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced decode.
        z: (batch, latent_dim), x_dec: (batch, seq_len)
        Returns logits: (batch, seq_len, vocab_size)
        """
        seq_len = x_dec.shape[1]
        emb = self.embed(x_dec)                             # (batch, seq, embed)
        z_exp = z.unsqueeze(1).expand(-1, seq_len, -1)      # (batch, seq, latent)
        inp = torch.cat([emb, z_exp], dim=-1)               # (batch, seq, embed+latent)
        h0 = torch.tanh(self.fc_decode(z)).unsqueeze(0)    # (1, batch, hidden)
        out, _ = self.decoder_gru(inp, h0)                  # (batch, seq, hidden)
        return self.fc_out(out)                             # (batch, seq, vocab)

    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor):
        mean, logvar = self.encode(x_enc)
        z = self.reparameterize(mean, logvar)
        logits = self.decode_tf(z, x_dec)
        return logits, mean, logvar


class _SELFIESDataset(Dataset):
    def __init__(self, data: list):
        self.data = data  # list of (x_enc, x_dec, x_tgt) integer lists

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_enc, x_dec, x_tgt = self.data[idx]
        return (
            torch.tensor(x_enc, dtype=torch.long),
            torch.tensor(x_dec, dtype=torch.long),
            torch.tensor(x_tgt, dtype=torch.long),
        )


class MolecularVAE:
    """
    SELFIES-based Variational Autoencoder for continuous molecular representation.

    Provides a continuous latent space for CMA-MAE optimization:
    - encode(smiles) → z ∈ ℝ^latent_dim
    - decode(z) → SMILES

    Uses SELFIES internally (not SMILES) to maximise validity of decoded molecules.
    SELFIES guarantees syntactically valid molecular strings across the entire latent
    space, so CMA-ES can explore freely without generating unparseable outputs.

    Usage:
        vae = MolecularVAE(latent_dim=64)
        vae.fit(smiles_list, epochs=50)
        z = vae.encode('c1ccccc1')
        smi = vae.decode(z)
        vae.save('vae.pt')
        vae2 = MolecularVAE.load('vae.pt')
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        embed_dim: int = 64,
        device: str = 'auto',
    ):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self._max_len: Optional[int] = None
        self.tokenizer: Optional[SELFIESTokenizer] = None
        self.model: Optional[_VAEModel] = None

        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

    def fit(
        self,
        smiles_list: List[str],
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        kl_anneal_epochs: int = 10,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train the VAE on a list of SMILES strings.

        Converts SMILES → SELFIES internally, builds tokenizer vocabulary,
        then trains using teacher forcing with KL annealing.

        Returns list of per-epoch average loss values.
        """
        import selfies as sf

        print(f"Converting {len(smiles_list)} SMILES to SELFIES...")
        selfies_list = []
        for smi in smiles_list:
            sel = _smiles_to_selfies(smi)
            if sel:
                selfies_list.append(sel)

        if not selfies_list:
            raise ValueError("No valid SELFIES found in training data.")
        print(f"  {len(selfies_list)}/{len(smiles_list)} SMILES converted to SELFIES")

        self.tokenizer = SELFIESTokenizer()
        self.tokenizer.fit(selfies_list)
        print(f"Vocabulary size: {self.tokenizer.vocab_size}")

        # max_len = longest tokenised SELFIES + SOS + EOS
        lengths = [len(list(sf.split_selfies(s))) + 2 for s in selfies_list]
        max_len = max(lengths)
        self._max_len = max_len
        print(f"Max sequence length: {max_len}")

        # Build (x_enc, x_dec, x_tgt) triples:
        #   x_enc = [SOS, tok1, ..., tokN, EOS, PAD, ...] length max_len
        #   x_dec = x_enc[:-1]  (decoder input, no last EOS)
        #   x_tgt = x_enc[1:]   (reconstruction target, no SOS)
        data = []
        for sel in selfies_list:
            x_enc = self.tokenizer.encode(sel, max_len=max_len)
            x_dec = x_enc[:-1]
            x_tgt = x_enc[1:]
            data.append((x_enc, x_dec, x_tgt))

        loader = DataLoader(_SELFIESDataset(data), batch_size=batch_size,
                            shuffle=True, drop_last=False)

        self.model = _VAEModel(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            pad_idx=self.tokenizer.pad_idx,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_idx)

        losses = []
        print(f"\nTraining VAE ({epochs} epochs, device={self.device})...")
        for epoch in range(epochs):
            self.model.train()
            # KL annealing: beta ramps 0 → 1 over kl_anneal_epochs
            beta = min(1.0, (epoch + 1) / max(1, kl_anneal_epochs))
            epoch_loss = 0.0
            n_batches = 0

            for x_enc, x_dec, x_tgt in loader:
                x_enc = x_enc.to(self.device)
                x_dec = x_dec.to(self.device)
                x_tgt = x_tgt.to(self.device)

                optimizer.zero_grad()
                logits, mean, logvar = self.model(x_enc, x_dec)

                # Reconstruction loss (cross-entropy over vocab)
                B, T, V = logits.shape
                recon = criterion(logits.reshape(-1, V), x_tgt.reshape(-1))

                # KL divergence regularisation
                kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

                loss = recon + beta * kl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / max(1, n_batches)
            losses.append(avg)
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"  Epoch {epoch+1:3d}/{epochs}: loss={avg:.4f}, beta={beta:.3f}")

        print(f"Training complete. Final loss: {losses[-1]:.4f}")
        return losses

    def encode(self, smiles: str) -> Optional[np.ndarray]:
        """
        Encode a SMILES string to a latent vector z (deterministic — returns mean).
        Returns numpy array shape (latent_dim,), or None on failure.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("VAE not trained. Call fit() or load() first.")
        sel = _smiles_to_selfies(smiles)
        if sel is None:
            return None
        indices = self.tokenizer.encode(sel, max_len=self._max_len)
        x = torch.tensor([indices], dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.no_grad():
            mean, _ = self.model.encode(x)
        return mean.cpu().numpy()[0]

    def decode(self, z: np.ndarray, max_len: Optional[int] = None) -> Optional[str]:
        """
        Decode a latent vector z to a SMILES string (greedy autoregressive).
        Returns canonical SMILES, or None if decoding fails.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("VAE not trained. Call fit() or load() first.")
        if max_len is None:
            max_len = self._max_len

        z_t = torch.tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, latent)
        h = torch.tanh(self.model.fc_decode(z_t)).unsqueeze(0)                        # (1, 1, hidden)
        token = torch.tensor([[self.tokenizer.sos_idx]], dtype=torch.long, device=self.device)

        self.model.eval()
        token_indices = []
        with torch.no_grad():
            for _ in range(max_len):
                emb = self.model.embed(token)                       # (1, 1, embed)
                inp = torch.cat([emb, z_t.unsqueeze(1)], dim=-1)   # (1, 1, embed+latent)
                out, h = self.model.decoder_gru(inp, h)             # (1, 1, hidden)
                logit = self.model.fc_out(out.squeeze(1))           # (1, vocab)
                next_tok = logit.argmax(-1, keepdim=True)           # (1, 1)
                idx = next_tok.item()
                if idx == self.tokenizer.eos_idx:
                    break
                if idx != self.tokenizer.pad_idx:
                    token_indices.append(idx)
                token = next_tok

        selfies_str = self.tokenizer.decode(token_indices)
        if not selfies_str:
            return None
        return _selfies_to_smiles(selfies_str)

    def save(self, path: str):
        """Save model weights, tokenizer, and config to a .pt checkpoint."""
        if self.model is None:
            raise RuntimeError("No model to save. Call fit() first.")
        torch.save({
            'model_state': self.model.state_dict(),
            'tokenizer': self.tokenizer.to_dict(),
            'config': {
                'latent_dim': self.latent_dim,
                'hidden_dim': self.hidden_dim,
                'embed_dim': self.embed_dim,
                'max_len': self._max_len,
            },
        }, path)
        print(f"VAE saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = 'auto') -> 'MolecularVAE':
        """Load a saved VAE checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        cfg = checkpoint['config']
        vae = cls(
            latent_dim=cfg['latent_dim'],
            hidden_dim=cfg['hidden_dim'],
            embed_dim=cfg['embed_dim'],
            device=device,
        )
        vae.tokenizer = SELFIESTokenizer.from_dict(checkpoint['tokenizer'])
        vae.model = _VAEModel(
            vocab_size=vae.tokenizer.vocab_size,
            embed_dim=vae.embed_dim,
            hidden_dim=vae.hidden_dim,
            latent_dim=vae.latent_dim,
            pad_idx=vae.tokenizer.pad_idx,
        ).to(vae.device)
        vae.model.load_state_dict(checkpoint['model_state'])
        vae.model.eval()
        vae._max_len = cfg['max_len']
        print(f"VAE loaded from {path} (latent_dim={vae.latent_dim}, device={vae.device})")
        return vae
