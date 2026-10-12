import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, bidir=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                            batch_first=True, bidirectional=bidir)
        self.classifier = nn.Linear(hidden_dim*(2 if bidir else 1), num_classes)

    def forward(self, x, lengths):
        emb = self.embedding(x)   
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), 
                                                batch_first=True, enforce_sorted=False)
        out_p, (h_n, c_n) = self.lstm(packed)
        # Handle bidirectional
        if self.lstm.bidirectional:
            last_forward = h_n[-2]
            last_backward = h_n[-1]
            last = torch.cat((last_forward, last_backward), dim=1)
        else:
            last = h_n[-1]
        return self.classifier(last)