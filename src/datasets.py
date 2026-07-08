"""PyTorch Dataset classes and collation/normalization utilities."""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# ── Normalization ─────────────────────────────────────────────────────────────

def compute_feature_stats(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std across all timesteps."""
    all_feats = np.concatenate(sequences, axis=0)
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def normalize_sequences(
    sequences: list[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> list[np.ndarray]:
    return [(s - mean) / std for s in sequences]


def normalize_sequences_per_session(
    sequences: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Standardize each sequence by its own per-feature mean/std.

    Fully local: removes both source- and user-level offsets without needing
    source labels. Used for Stage 2 Variant B, where a composed session mixes
    atomics from several sources so per-source statistics are ill-defined.
    """
    out = []
    for s in sequences:
        mean = s.mean(axis=0)
        std = s.std(axis=0)
        std[std < 1e-8] = 1.0
        out.append((s - mean) / std)
    return out


def normalize_sequences_per_source(
    sequences: list[np.ndarray],
    sources: np.ndarray,
) -> list[np.ndarray]:
    """
    Standardize each sequence using statistics computed within its own source.

    Removes source-specific feature offsets/scales so the model sees each
    sequence relative to its source baseline (Option A, transductive: the
    held-out / test source self-normalizes from its own sequences). No labels
    are used, so this is leakage-free.
    """
    sources = np.asarray(sources)
    out: list[np.ndarray] = [None] * len(sequences)
    for s in np.unique(sources):
        idx = np.where(sources == s)[0]
        feats = np.concatenate([sequences[i] for i in idx], axis=0)
        mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        std[std < 1e-8] = 1.0
        for i in idx:
            out[i] = (sequences[i] - mean) / std
    return out


# ── Stage 1 Datasets ─────────────────────────────────────────────────────────

class HoverSequenceDataset(Dataset):
    """Variable-length hover sequences with a single label per sequence."""

    def __init__(self, sequences: list[np.ndarray], labels: np.ndarray):
        self.seqs = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.lengths = torch.tensor([s.shape[0] for s in sequences])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx], self.lengths[idx]


class AugmentableDataset(Dataset):
    """
    Hover sequence dataset with optional random subsequence augmentation.
    When in training mode, returns a random contiguous subsequence of at
    least augment_min_frac * original_length timesteps.
    """

    def __init__(
        self,
        sequences: list[np.ndarray],
        labels: np.ndarray,
        augment_min_frac: float = 0.7,
    ):
        self.seqs = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment_min_frac = augment_min_frac
        self._training = False

    def train_mode(self):
        self._training = True

    def eval_mode(self):
        self._training = False

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        if self._training and self.augment_min_frac < 1.0:
            T = seq.shape[0]
            min_len = max(1, int(T * self.augment_min_frac))
            sub_len = torch.randint(min_len, T + 1, (1,)).item()
            start = torch.randint(0, T - sub_len + 1, (1,)).item()
            seq = seq[start : start + sub_len]
        return seq, self.labels[idx], torch.tensor(seq.shape[0])


def collate_sequences(batch):
    """Pad and sort by length descending for pack_padded_sequence."""
    seqs, labels, lengths = zip(*batch)
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    seqs = [seqs[i] for i in order]
    labels = torch.stack([labels[i] for i in order])
    lengths = torch.stack([lengths[i] for i in order])
    return pad_sequence(seqs, batch_first=True), labels, lengths


# ── Stage 2 Datasets ─────────────────────────────────────────────────────────

class TimestepLabelDataset(Dataset):
    """Variable-length sessions with per-timestep labels."""

    def __init__(self, sequences: list[np.ndarray], labels: list[np.ndarray]):
        self.seqs = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.labels = [torch.tensor(l, dtype=torch.long) for l in labels]
        self.lengths = torch.tensor([s.shape[0] for s in sequences])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx], self.lengths[idx]


def collate_timesteps(batch):
    """Pad sequences and labels, sort by length descending. Labels padded with -1."""
    seqs, labels, lengths = zip(*batch)
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    seqs = [seqs[i] for i in order]
    labels = [labels[i] for i in order]
    lengths = torch.stack([lengths[i] for i in order])
    padded_seqs = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    return padded_seqs, padded_labels, lengths
