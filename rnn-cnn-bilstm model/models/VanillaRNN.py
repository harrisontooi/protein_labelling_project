import torch
import torch.nn as nn


class VanillaRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers, 
                          batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        # x: (B, L), lengths: (B,)
        emb = self.embedding(x)       # (B, L, E)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), 
                                                   batch_first=True, enforce_sorted=False)
        out_p, h_n = self.rnn(packed)
        # h_n: (num_layers, B, hidden_dim)
        last = h_n[-1]                # (B, hidden_dim)
        return self.classifier(last)
