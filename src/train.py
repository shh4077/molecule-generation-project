from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tokenizer import SmilesTokenizer


DATA_PATH = Path("data/train.txt")

MAX_SAMPLES = 10_000
MAX_LENGTH = 64
BATCH_SIZE = 32

EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2

LEARNING_RATE = 1e-3
EPOCHS = 3


def load_smiles(path: Path) -> list[str]:
    df = pd.read_csv(path)

    return (
        df["SMILES"]
        .dropna()
        .astype(str)
        .head(MAX_SAMPLES)
        .tolist()
    )


class SmilesDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.stoi["<PAD>"]

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]

        token_ids = self.tokenizer.encode(smiles)
        token_ids = token_ids[:self.max_length]

        padding_needed = self.max_length - len(token_ids)
        token_ids += [self.pad_token_id] * padding_needed

        x = torch.tensor(token_ids[:-1], dtype=torch.long)
        y = torch.tensor(token_ids[1:], dtype=torch.long)

        return x, y


class SmilesTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_length):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_length = x.shape

        positions = torch.arange(seq_length, device=x.device)
        positions = positions.unsqueeze(0)

        x = self.token_embedding(x) + self.position_embedding(positions)

        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=x.device),
            diagonal=1,
        ).bool()

        x = self.transformer(x, mask=causal_mask)

        logits = self.output_layer(x)

        return logits


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()

    total_loss = 0.0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    smiles_list = load_smiles(DATA_PATH)
    tokenizer = SmilesTokenizer(smiles_list)

    dataset = SmilesDataset(
        smiles_list=smiles_list,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = SmilesTransformer(
        vocab_size=len(tokenizer.tokens),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_length=MAX_LENGTH,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.stoi["<PAD>"],
    )

    for epoch in range(EPOCHS):
        loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss:.4f}")

    torch.save(
        model.state_dict(),
        "smiles_transformer.pt",
    )

    print("Model saved: smiles_transformer.pt")


if __name__ == "__main__":
    main()