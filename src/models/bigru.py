"""Stage 1 BiGRU models for sequence-level classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class BiGRUClassifier(nn.Module):
    """Single-layer BiGRU → concat final hidden → dropout → linear."""

    def __init__(self, input_dim=24, hidden_dim=64, num_classes=7, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, h = self.gru(packed)
        out = self.drop(torch.cat([h[0], h[1]], dim=1))
        return self.fc(out)

    def forward_proba(self, x, lengths):
        return F.softmax(self.forward(x, lengths), dim=1)


class BiGRUAttentionClassifier(nn.Module):
    """BiGRU with attention pooling over all hidden states."""

    def __init__(self, input_dim=24, hidden_dim=64, num_classes=7, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.gru(packed)
        from torch.nn.utils.rnn import pad_packed_sequence
        output, _ = pad_packed_sequence(packed_out, batch_first=True)

        mask = torch.arange(output.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = self.attn(output).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)

        context = (output * weights).sum(dim=1)
        return self.fc(self.drop(context))
