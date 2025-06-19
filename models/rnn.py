from typing import Literal, Optional

import torch
import torch.nn as nn

from utils.constants import MAX_LEN


class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 pooling: Optional[Literal['mean', 'max']] = None):
        super(RNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        i2h = [nn.Linear(input_size, hidden_size, bias=False)]
        h2h = [nn.Linear(hidden_size, hidden_size)]

        for i in range(1, num_layers):
            i2h.append(nn.Linear(hidden_size, hidden_size, bias=False))
            h2h.append(nn.Linear(hidden_size, hidden_size))

        self.i2h = nn.ModuleList(i2h)
        self.h2h = nn.ModuleList(h2h)

        self.pooling = pooling

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=x.device)

        h_t_minus_1 = hx
        hidden_states = []

        for layer in range(self.num_layers):
            for t in range(seq_len):
                input_t = x[:, t, :] if layer == 0 else hidden_states[((layer - 1) * seq_len) + t]
                h_out = self.i2h[layer](input_t) + self.h2h[layer](h_t_minus_1)
                hidden_states.append(h_out)
                h_t_minus_1 = h_out

        # last hidden state from last layer
        return hidden_states[-1]


class RNNTextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, input_size: int, hidden_size: int,
                 num_classes: int, num_layers: int = 1,
                 bidirectional: bool = False):
        super(RNNTextClassifier, self).__init__()

        self.word_embeds = nn.Embedding(vocab_size, embed_dim)

        self.l2r_rnn = RNN(input_size, hidden_size, num_layers=num_layers)
        self.r2l_rnn = None

        if bidirectional:
            self.r2l_rnn = RNN(input_size, hidden_size, num_layers=num_layers)

        hidden_size = hidden_size * 2 if bidirectional else hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.word_embeds(x)

        h_t = self.l2r_rnn(x, hx)

        if self.r2l_rnn is not None:
            h_t_rev = self.r2l_rnn(x.flip((2,)), hx)
            h_t = torch.cat((h_t_rev, h_t), dim=-1)

        return self.classifier(h_t)
