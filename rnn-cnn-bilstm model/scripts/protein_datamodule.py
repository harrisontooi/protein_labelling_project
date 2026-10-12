import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class ProteinSequenceDataset(Dataset):
    def __init__(self, df, aa2idx, label2idx,
                 seq_col="sequence", label_col="location"):
        self.seqs   = df[seq_col].tolist()
        self.labels = [ label2idx[l] for l in df[label_col].tolist() ]
        self.aa2idx = aa2idx
        self.X_idx  = aa2idx["X"]

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, i):
        seq = self.seqs[i]
        lbl = self.labels[i]
        # numericalize
        toks = torch.tensor(
            [ self.aa2idx.get(aa, self.X_idx) for aa in seq ],
            dtype=torch.long
        )
        return toks, lbl

class ProteinDataModule:
    def __init__(self,
                 csv_path,
                 batch_size=64,
                 val_frac=0.2,
                 random_state=42):
        self.csv_path    = csv_path
        self.batch_size  = batch_size
        self.val_frac    = val_frac
        self.random_state= random_state

        # build vocab: 20 AAs + X + <pad> and create mapping
        VOCAB = list("ACDEFGHIKLMNPQRSTVWY") + ["X","<pad>"]
        self.aa2idx = { aa:i for i,aa in enumerate(VOCAB) }
        self.pad_idx = self.aa2idx["<pad>"]

        # load & split
        df = pd.read_csv(self.csv_path)
        df = (df.dropna(subset=["sequence","location"])
                 .drop_duplicates())
        labels = sorted(df["location"].unique().tolist())
        self.label2idx = { l:i for i,l in enumerate(labels) }

        train_df, val_df = train_test_split(
            df,
            test_size   = self.val_frac,
            stratify   = df["location"],
            random_state=self.random_state
        )

        # make datasets
        self.train_ds = ProteinSequenceDataset(train_df,
                            aa2idx=self.aa2idx,
                            label2idx=self.label2idx)
        self.val_ds   = ProteinSequenceDataset(val_df,
                            aa2idx=self.aa2idx,
                            label2idx=self.label2idx)

    @property
    def vocab(self):
        return self.aa2idx

    @property
    def label_mappings(self):
        return self.label2idx

    def collate_fn(self, batch):
        # get sequences and labels
        seqs, labs = zip(*batch)
        # calculate original lengths (before padding)
        lengths = torch.tensor([len(seq) for seq in seqs], dtype=torch.long)
        # pad sequences
        seqs = pad_sequence(seqs, batch_first=True, padding_value=self.pad_idx)
        labs = torch.tensor(labs, dtype=torch.long)
        return seqs, labs, lengths  # returns (padded_seqs, labels, lengths)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self.collate_fn,
                          num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=self.collate_fn,
                          num_workers=4)
