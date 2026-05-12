from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tokenizer import SmilesTokenizer


DATA_PATH = Path("data/train.txt")

MAX_SAMPLES = 10000
MAX_LENGTH = 64

BATCH_SIZE = 4

EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2


def load_smiles(path: Path) -> list[str]:
    df = pd.read_csv(path)

    smiles_list = (
        df["SMILES"]
        .dropna()
        .astype(str)
        .head(MAX_SAMPLES)
        .tolist()
    )

    return smiles_list


class SmilesDataset(Dataset):
    def __init__(
        self,
        smiles_list,
        tokenizer,
        max_length,
    ):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.pad_token_id = tokenizer.stoi["<PAD>"]

    def __len__(self):
        return len(self.smiles_list)

    def pad_sequence(self, token_ids):
        token_ids = token_ids[:self.max_length]

        padding_needed = (
            self.max_length
            - len(token_ids)
        )

        token_ids += (
            [self.pad_token_id]
            * padding_needed
        )

        return token_ids

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]

        token_ids = self.tokenizer.encode(smiles)

        token_ids = self.pad_sequence(token_ids)

        x = torch.tensor(
            token_ids[:-1],
            dtype=torch.long,
        )

        y = torch.tensor(
            token_ids[1:],
            dtype=torch.long,
        )

        return x, y


class SmilesTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        max_length,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
        )

        self.position_embedding = nn.Embedding(
            max_length,
            embed_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_layer = nn.Linear(
            embed_dim,
            vocab_size,
        )

    def forward(self, x):
        batch_size, seq_length = x.shape

        positions = torch.arange(
            seq_length,
            device=x.device,
        )

        positions = positions.unsqueeze(0)

        token_embeddings = self.token_embedding(x)

        position_embeddings = (
            self.position_embedding(positions)
        )

        x = (
            token_embeddings
            + position_embeddings
        )

        x = self.transformer(x)

        logits = self.output_layer(x)

        return logits


def main():
    smiles_list = load_smiles(DATA_PATH)

    tokenizer = SmilesTokenizer(smiles_list)

    dataset = SmilesDataset(
        smiles_list,
        tokenizer,
        MAX_LENGTH,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    x, y = next(iter(dataloader))

    model = SmilesTransformer(
        vocab_size=len(tokenizer.tokens),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_length=MAX_LENGTH,
    )

    logits = model(x)

    print("Input shape:")
    print(x.shape)
    print()

    print("Output logits shape:")
    print(logits.shape)
    print()

    print("Vocabulary size:")
    print(len(tokenizer.tokens))


if __name__ == "__main__":
    main()