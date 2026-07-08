"""Stage 2 models for per-timestep task prediction."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from TorchCRF import CRF

import numpy as np


class UniGRU_CRF(nn.Module):
    """Forward-only GRU with CRF for per-timestep labeling (online-capable)."""

    def __init__(self, input_dim, hidden_dim=64, num_classes=7, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.crf = CRF(num_classes, batch_first=True)

    def _get_emissions(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.gru(packed)
        output, _ = pad_packed_sequence(packed_out, batch_first=True)
        return self.fc(self.dropout(output))

    def forward(self, x, labels, lengths):
        """Compute negative log-likelihood loss."""
        emissions = self._get_emissions(x, lengths)
        mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        safe_labels = labels.clone()
        safe_labels[safe_labels == -1] = 0
        return -self.crf(emissions, safe_labels, mask=mask, reduction="mean")

    @torch.no_grad()
    def decode(self, x, lengths):
        emissions = self._get_emissions(x, lengths)
        mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        return self.crf.decode(emissions, mask=mask)


class BiGRU_CRF(nn.Module):
    """Bidirectional GRU with CRF for per-timestep labeling."""

    def __init__(self, input_dim, hidden_dim=64, num_classes=7, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.crf = CRF(num_classes, batch_first=True)

    def _get_emissions(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.gru(packed)
        output, _ = pad_packed_sequence(packed_out, batch_first=True)
        return self.fc(self.dropout(output))

    def forward(self, x, labels, lengths):
        emissions = self._get_emissions(x, lengths)
        mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        safe_labels = labels.clone()
        safe_labels[safe_labels == -1] = 0
        return -self.crf(emissions, safe_labels, mask=mask, reduction="mean")

    @torch.no_grad()
    def decode(self, x, lengths):
        emissions = self._get_emissions(x, lengths)
        mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        return self.crf.decode(emissions, mask=mask)


class BiGRU_Timestep(nn.Module):
    """Bidirectional GRU without CRF (ablation)."""

    def __init__(self, input_dim, hidden_dim=64, num_classes=7, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return self.fc(self.dropout(output))


class UniGRU_Timestep(nn.Module):
    """Forward-only GRU without CRF (online ablation)."""

    def __init__(self, input_dim, hidden_dim=64, num_classes=7, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return self.fc(self.dropout(output))
