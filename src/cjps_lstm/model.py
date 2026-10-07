"""PyTorch model for next-touchpoint prediction.

Architecture::

    per event:  Embedding(type_touch) ⊕ [duration, gap, device, purchases]
                          │
                    LSTM (packed, padding-aware)
                          │
                  last valid hidden state
                          │
        ⊕ static vector (demographics + behavioural aggregates)
                          │
                  Dense → Dropout → Linear
                          │
                logits over touchpoint classes

Why this shape:

* **Embedding, not one-hot, for touchpoints.** There are 20 touchpoint
  codes; one-hot would give a sparse 20-dim vector with no shared
  structure, while an embedding lets the network place similar channels
  near each other. 20 categories is small, so the embedding is more about
  a learned metric space than compression.
* **pack_padded_sequence.** Padded zeros must not advance the LSTM, or a
  user with 3 real events padded to length 132 would be summarised by the
  state after 129 steps of silence. Packing enforces that; the hidden
  state is taken at each sequence's true final step.
* **Static features bypass the LSTM.** Demographics are constant per user;
  repeating them at every timestep would add 128 redundant copies of the
  same vector. They are concatenated to the final hidden state instead —
  the standard static+sequence fusion pattern.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset


class JourneyDataset(Dataset):
    """Tensors for one partition: padded sequence, static vector, label."""

    def __init__(
        self,
        seq_ids: np.ndarray,
        seq_num: np.ndarray,
        lengths: np.ndarray,
        static: np.ndarray,
        labels: np.ndarray,
    ):
        self.seq_ids = torch.as_tensor(seq_ids, dtype=torch.long)
        self.seq_num = torch.as_tensor(seq_num, dtype=torch.float32)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long)
        self.static = torch.as_tensor(static, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return (
            self.seq_ids[i],
            self.seq_num[i],
            self.lengths[i],
            self.static[i],
            self.labels[i],
        )


class JourneyLSTM(nn.Module):
    """LSTM classifier over touchpoint sequences plus static features."""

    def __init__(
        self,
        vocab_size: int,
        n_event_features: int,
        n_static: int,
        n_classes: int,
        emb_dim: int = 32,
        lstm_units: int = 64,
        lstm_layers: int = 1,
        dense_units: int = 64,
        dropout: float = 0.2,
        bidirectional: bool = False,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, emb_dim, padding_idx=padding_idx
        )
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=emb_dim + n_event_features,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        head_in = lstm_units * direction_factor + n_static
        self.head = nn.Sequential(
            nn.Linear(head_in, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, n_classes),
        )

    def forward(self, seq_ids, seq_num, lengths, static):
        emb = self.embedding(seq_ids)
        x = torch.cat([emb, seq_num], dim=-1)

        # clamp(min=1): a length-0 sequence would make pack raise; every real
        # example has >= 2 events, this only protects degenerate inputs.
        packed = pack_padded_sequence(
            x, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)

        if self.bidirectional:
            # h_n: (layers*2, N, units) — concat the two final directions.
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]

        return self.head(torch.cat([h_last, static], dim=-1))
